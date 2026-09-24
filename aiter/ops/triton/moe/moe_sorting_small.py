"""One-launch moe_sorting for decode row counts (see aiter.fused_moe.moe_sorting, which
routes here when AITER_USE_SMALL_M_MOE_SORTING=1)."""

from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_sorting_small import (
    _moe_sorting_small_kernel,
)

# above this many rows the multi-phase Opus sorting kernel wins
MOE_SORTING_SMALL_MAX_TOKENS = 64

# masks are static per layer: (data_ptr, numel, device) -> (local ids, count, mask kept alive)
_LOCAL_EXPERT_IDS: dict = {}


def cached_local_expert_ids(
    expert_mask: Optional[torch.Tensor], num_experts: int, device
) -> Optional[Tuple[torch.Tensor, int]]:
    """(local ids, local expert count) for a mask, cached by storage. None during graph
    capture for a mask not seen yet: counting it is a device sync."""
    if expert_mask is None:
        key = (None, num_experts, torch.device(device).index)
    else:
        key = (expert_mask.data_ptr(), expert_mask.numel(), expert_mask.device.index)
    entry = _LOCAL_EXPERT_IDS.get(key)
    if entry is None:
        if expert_mask is None:
            num_local = num_experts
        elif torch.cuda.is_current_stream_capturing():
            return None
        else:
            num_local = int(expert_mask.count_nonzero().item())
        entry = (
            local_expert_ids_from_mask(expert_mask, num_experts, device),
            num_local,
            expert_mask,
        )
        _LOCAL_EXPERT_IDS[key] = entry
    return entry[0], entry[1]


def local_expert_ids_from_mask(
    expert_mask: Optional[torch.Tensor], num_experts: int, device
) -> torch.Tensor:
    """aiter's local expert numbering: the running count of the mask, -1 elsewhere."""
    if expert_mask is None:
        return torch.arange(num_experts, dtype=torch.int32, device=device)
    mask = expert_mask.to(torch.int32)
    local = torch.cumsum(mask, 0, dtype=torch.int32) - 1
    return torch.where(mask != 0, local, torch.full_like(local, -1)).contiguous()


def moe_sorting_small(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_ids: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    model_dim: int,
    moe_buf_dtype: torch.dtype,
    block_size: int,
    zero_moe_buf: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """aiter.fused_moe.moe_sorting in one launch: (sorted_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, moe_buf) with aiter's shapes, moe_buf empty unless
    zero_moe_buf; rows at and past num_token_non_padded become (expert 0, weight 0)."""
    M, topk = topk_ids.shape
    if M > MOE_SORTING_SMALL_MAX_TOKENS:
        raise ValueError(
            f"moe_sorting_small takes at most {MOE_SORTING_SMALL_MAX_TOKENS} rows, got {M}"
        )
    if block_size & (block_size - 1) or block_size <= 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise TypeError("topk_ids must be a contiguous int32 tensor")
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        raise TypeError("topk_weights must be a contiguous float32 tensor")
    if local_expert_ids.dtype != torch.int32 or local_expert_ids.numel() != num_experts:
        raise TypeError("local_expert_ids must be int32 with one entry per expert")
    device = topk_ids.device
    max_num_tokens_padded = M * topk + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if zero_moe_buf:
        moe_buf = torch.empty((M, model_dim), dtype=moe_buf_dtype, device=device)
    else:
        moe_buf = torch.empty((0, 0), dtype=moe_buf_dtype, device=device)

    e_pad = triton.next_power_of_2(num_local_experts + 1)
    ec = min(4, e_pad)
    num_ec = e_pad // ec
    m_pad = max(2, triton.next_power_of_2(M))
    _moe_sorting_small_kernel[(num_ec + M,)](
        topk_ids,
        topk_weights,
        local_expert_ids,
        num_token_non_padded if num_token_non_padded is not None else num_valid_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        M,
        model_dim,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        M_PAD=m_pad,
        E_PAD=e_pad,
        EC=ec,
        NUM_EC=num_ec,
        BLOCK_M=block_size,
        NB_PER_E=max(2, triton.next_power_of_2((M + block_size - 1) // block_size)),
        HAS_PAD_COUNT=num_token_non_padded is not None,
        ZERO_MOE_BUF=zero_moe_buf,
        BLOCK_D=1024,
        num_warps=1 if M <= 8 else 4,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

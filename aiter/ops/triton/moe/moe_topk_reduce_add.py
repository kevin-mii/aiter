"""The FlyDSL stage2 top-k reduction with the shared-expert add in the same launch.

Inside `with fused_moe_reduce_shared_add(shared_output, alpha):` the dense bf16/fp16
reduction of fused_moe writes alpha * routed + shared_output into its output; the
returned request says whether it fired (otherwise add shared_output yourself)."""

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_topk_reduce_add import (
    _moe_topk_reduce_add_kernel,
)


@dataclass
class FusedReduceRequest:
    shared_output: torch.Tensor
    alpha: float
    fired: bool = False


_REQUEST: contextvars.ContextVar[Optional[FusedReduceRequest]] = contextvars.ContextVar(
    "aiter_fused_moe_reduce_shared_add", default=None
)


@contextlib.contextmanager
def fused_moe_reduce_shared_add(shared_output: torch.Tensor, alpha: float = 1.0):
    request = FusedReduceRequest(shared_output, float(alpha))
    token = _REQUEST.set(request)
    try:
        yield request
    finally:
        _REQUEST.reset(token)


def try_reduce_shared_add(
    target,
    out,
    token_num,
    topk,
    model_dim,
    expert_mask,
    topk_ids,
    stream,
    is_fp8,
    topk_weights,
) -> bool:
    """The reduction of _run_moe_reduction with the pending shared-expert add, when a
    fused_moe_reduce_shared_add scope is active and the call is one it serves."""
    request = _REQUEST.get()
    if request is None:
        return False
    shared = request.shared_output
    if not (
        not is_fp8
        and topk_weights is None
        and stream is None
        and out.dtype in (torch.bfloat16, torch.float16)
        and target.dtype == out.dtype == shared.dtype
        and tuple(out.shape) == (token_num, model_dim) == tuple(shared.shape)
        and target.numel() == token_num * topk * model_dim
        and (expert_mask is None or topk_ids is not None)
    ):
        return False
    moe_topk_reduce_add(
        target, shared, out, int(topk), topk_ids, expert_mask, alpha=request.alpha
    )
    request.fired = True
    return True


def moe_topk_reduce_add(
    per_slot: torch.Tensor,
    shared: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    topk_ids: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> None:
    """out[t] = alpha * sum_k per_slot[t * topk + k] + shared[t], skipping slots whose expert
    topk_ids maps outside expert_mask; fp32 accumulation rounded once to out's dtype."""
    M, D = out.shape
    assert shared.shape == (M, D), (shared.shape, out.shape)
    assert per_slot.numel() == M * topk * D, (per_slot.shape, M, topk, D)
    assert out.stride(1) == 1 and shared.stride(1) == 1
    x = per_slot.view(M * topk, D) if per_slot.dim() != 2 else per_slot
    assert x.stride(1) == 1
    has_mask = expert_mask is not None
    if has_mask:
        assert topk_ids is not None and topk_ids.shape == (M, topk)
        topk_ids = topk_ids.contiguous()
        expert_mask = expert_mask.contiguous()
    if M == 0:
        return
    block_d = min(1024, triton.next_power_of_2(D))
    _moe_topk_reduce_add_kernel[(M, triton.cdiv(D, block_d))](
        x,
        shared,
        out,
        topk_ids if has_mask else out,
        expert_mask if has_mask else out,
        D,
        x.stride(0),
        shared.stride(0),
        out.stride(0),
        float(alpha),
        TOPK=topk,
        HAS_MASK=has_mask,
        HAS_ALPHA=float(alpha) != 1.0,
        BLOCK_D=block_d,
        num_warps=4,
    )

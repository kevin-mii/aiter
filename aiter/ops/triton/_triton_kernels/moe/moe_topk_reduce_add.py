"""MoE top-k reduction with the shared-expert add folded in: fp32 over the slots in
ascending order as the FlyDSL moe_reduction kernel sums them, one rounding at the end.
"""

import triton
import triton.language as tl


@triton.jit
def _moe_topk_reduce_add_kernel(
    x_ptr,  # [M * TOPK, D] per-slot expert outputs
    shared_ptr,  # [M, D]
    out_ptr,  # [M, D]
    topk_ids_ptr,  # [M, TOPK] (HAS_MASK)
    expert_mask_ptr,  # [num_experts] (HAS_MASK): nonzero = local expert
    D,
    stride_xm,
    stride_sm,
    stride_om,
    alpha,
    TOPK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        v = tl.load(
            x_ptr + (row * TOPK + k) * stride_xm + offs, mask=mask, other=0.0
        ).to(tl.float32)
        if HAS_MASK:
            # rows of experts this rank does not hold were never written
            e = tl.load(topk_ids_ptr + row * TOPK + k)
            local = tl.load(expert_mask_ptr + e) != 0
            v = tl.where(local, v, 0.0)
        acc += v
    if HAS_ALPHA:
        acc *= alpha
    acc += tl.load(shared_ptr + row * stride_sm + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        out_ptr + row * stride_om + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
    )

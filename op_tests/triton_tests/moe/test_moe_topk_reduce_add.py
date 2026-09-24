"""moe_topk_reduce_add sums the top-k slots in ascending order in fp32 and adds the shared
expert with one rounding; inside fused_moe_reduce_shared_add the FlyDSL reduction uses it.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_topk_reduce_add import (
    fused_moe_reduce_shared_add,
    moe_topk_reduce_add,
)

E, TOPK, D = 64, 6, 5120


def _inputs(m, seed, local_fraction=0.25):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m * TOPK, D, device="cuda", dtype=torch.bfloat16, generator=g)
    shared = torch.randn(m, D, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.randint(0, E, (m, TOPK), device="cuda", dtype=torch.int32, generator=g)
    mask = (torch.arange(E, device="cuda") < int(E * local_fraction)).to(torch.int32)
    return x, shared, ids, mask


def _reference(x, shared, ids, mask, alpha):
    xs = x.view(shared.shape[0], TOPK, D).float()
    acc = torch.zeros_like(shared, dtype=torch.float32)
    for k in range(TOPK):
        v = xs[:, k]
        if mask is not None:
            v = torch.where((mask[ids[:, k]] != 0)[:, None], v, 0.0)
        acc = acc + v
    return (acc * alpha + shared.float()).to(shared.dtype)


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("use_mask", [True, False])
def test_matches_reference(m, use_mask):
    x, shared, ids, mask = _inputs(m, m)
    out = torch.empty_like(shared)
    moe_topk_reduce_add(
        x, shared, out, TOPK, ids if use_mask else None, mask if use_mask else None, 2.5
    )
    assert torch.equal(out, _reference(x, shared, ids, mask if use_mask else None, 2.5))


def test_rows_are_batch_invariant():
    x, shared, ids, mask = _inputs(300, 7)
    full = torch.empty_like(shared)
    moe_topk_reduce_add(x, shared, full, TOPK, ids, mask)
    idx = torch.arange(0, 300, 7, device="cuda")
    sub = torch.empty(len(idx), D, device="cuda", dtype=torch.bfloat16)
    moe_topk_reduce_add(
        x.view(300, TOPK, D)[idx].reshape(-1, D).contiguous(),
        shared[idx].contiguous(),
        sub,
        TOPK,
        ids[idx].contiguous(),
        mask,
    )
    assert torch.equal(sub, full[idx])


def test_flydsl_reduction_adds_the_shared_expert_inside_the_scope():
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    x, shared, ids, mask = _inputs(4, 11)
    out = torch.empty_like(shared)
    with fused_moe_reduce_shared_add(shared, 1.5) as request:
        _run_moe_reduction(x, out, 4, TOPK, D, mask, ids)
    assert request.fired
    assert torch.equal(out, _reference(x, shared, ids, mask, 1.5))

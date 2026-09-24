"""moe_sorting_small must reproduce aiter.fused_moe.moe_sorting bit for bit (the region the
MoE GEMMs read), and moe_sorting must route decode batches to it when enabled."""

import pytest
import torch

import aiter.fused_moe as fused_moe
from aiter.ops.triton.moe.moe_sorting_small import (
    local_expert_ids_from_mask,
    moe_sorting_small,
)

NUM_EXPERTS = 384
MODEL_DIM = 5120
TOPK = 6


def _routing(num_tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device="cuda", generator=g)[:TOPK]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    return ids, torch.rand(num_tokens, TOPK, device="cuda", generator=g)


def _mask(num_local, rank):
    mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cuda")
    mask[rank * num_local : (rank + 1) * num_local] = 1
    return mask


def _reference(ids, weights, block_size, mask):
    return fused_moe.moe_sorting(
        ids, weights, NUM_EXPERTS, MODEL_DIM, torch.bfloat16, block_size, mask, None, 0
    )


def _assert_same_sort(ref, out, block_size, num_valid_rows, num_tokens):
    """Compare what the GEMMs read: up to the padded total. Slots of padded rows race in
    aiter's kernel, so only their token part is compared."""
    assert torch.equal(ref[3], out[3]), "num_valid_ids"
    total = int(ref[3][0])
    ref_ids, out_ids = ref[0][:total], out[0][:total]
    tokens = ref_ids & 0xFFFFFF
    assert torch.equal(tokens, out_ids & 0xFFFFFF), "sorted tokens"
    racy_slot = (tokens >= num_valid_rows) & (tokens < num_tokens)
    assert torch.equal(
        torch.where(racy_slot, 0, ref_ids), torch.where(racy_slot, 0, out_ids)
    ), "sorted slots"
    assert torch.equal(ref[1][:total], out[1][:total]), "sorted weights"
    blocks = total // block_size
    assert torch.equal(ref[2][:blocks], out[2][:blocks]), "sorted expert ids"
    assert ref[4].shape == out[4].shape and torch.equal(ref[4], out[4]), "moe_buf"


@pytest.mark.parametrize(
    "rank, num_tokens, block_size", [(0, 1, 64), (3, 64, 16)], ids=["first", "last"]
)
def test_matches_aiter_expert_parallel(rank, num_tokens, block_size):
    mask = _mask(96, rank)
    ids, weights = _routing(num_tokens, num_tokens)
    ref = _reference(ids, weights, block_size, mask)
    out = moe_sorting_small(
        ids,
        weights,
        local_expert_ids_from_mask(mask, NUM_EXPERTS, "cuda"),
        96,
        NUM_EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        block_size,
        zero_moe_buf=True,
    )
    _assert_same_sort(ref, out, block_size, num_tokens, num_tokens)


@pytest.mark.parametrize("num_valid", [1, 64])
def test_padded_rows_are_filled_in_the_same_launch(num_valid):
    mask = _mask(96, 1)
    ids, weights = _routing(64, 7 * 64 + num_valid)
    masked_ids, masked_weights = ids.clone(), weights.clone()
    masked_ids[num_valid:] = 0
    masked_weights[num_valid:] = 0.0
    ref = _reference(masked_ids, masked_weights, 32, mask)
    out = moe_sorting_small(
        ids,
        weights,
        local_expert_ids_from_mask(mask, NUM_EXPERTS, "cuda"),
        96,
        NUM_EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        32,
        zero_moe_buf=True,
        num_token_non_padded=torch.tensor(
            [num_valid], dtype=torch.int32, device="cuda"
        ),
    )
    _assert_same_sort(ref, out, 32, num_valid, 64)
    assert torch.equal(ids, masked_ids) and torch.equal(weights, masked_weights)


def test_moe_sorting_routes_small_batches_when_enabled(monkeypatch):
    mask = _mask(96, 2)
    ids, weights = _routing(8, 3)
    ref = _reference(ids, weights, 32, mask)
    calls = []
    import aiter.ops.triton.moe.moe_sorting_small as small

    monkeypatch.setattr(fused_moe, "_USE_SMALL_M_MOE_SORTING", True)
    original = small.moe_sorting_small
    monkeypatch.setattr(
        small, "moe_sorting_small", lambda *a, **k: calls.append(1) or original(*a, **k)
    )
    out = fused_moe.moe_sorting(
        ids, weights, NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 32, mask
    )
    assert calls, "the small-M sort did not serve the call"
    _assert_same_sort(ref, out, 32, 8, 8)

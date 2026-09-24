"""moe_sorting in one Triton launch for decode row counts: the five outputs of
aiter.fused_moe.moe_sorting bit for bit, with an optional padded-row fill."""

import triton
import triton.language as tl


@triton.jit
def _moe_sorting_small_kernel(
    topk_ids_ptr,  # [M, TOPK_IN] int32 global expert ids
    topk_weights_ptr,  # [M, TOPK_IN] fp32
    local_expert_ids_ptr,  # [num_experts] int32: local index, -1 when not local
    num_token_non_padded_ptr,  # [1] int32 (read when HAS_PAD_COUNT)
    sorted_ids_ptr,  # [max_padded] int32
    sorted_weights_ptr,  # [max_padded] fp32
    sorted_expert_ids_ptr,  # [max_blocks] int32
    num_valid_ids_ptr,  # [2] int32
    moe_buf_ptr,  # [M, model_dim] (zeroed when ZERO_MOE_BUF)
    num_tokens,
    model_dim,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    M_PAD: tl.constexpr,
    E_PAD: tl.constexpr,  # local expert bins, > number of local experts
    EC: tl.constexpr,  # local experts per sorting program
    NUM_EC: tl.constexpr,  # sorting programs (E_PAD // EC)
    BLOCK_M: tl.constexpr,  # aiter block_m: rows per GEMM tile
    NB_PER_E: tl.constexpr,  # >= max blocks one expert can fill
    HAS_PAD_COUNT: tl.constexpr,
    ZERO_MOE_BUF: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if HAS_PAD_COUNT:
        n_valid = tl.load(num_token_non_padded_ptr)
    else:
        n_valid = num_tokens

    if pid >= NUM_EC:
        # programs NUM_EC.. zero one output row each and mask padded top-k rows (expert 0, weight 0)
        row = pid - NUM_EC
        if ZERO_MOE_BUF:
            offs_d = tl.arange(0, BLOCK_D)
            for d0 in range(0, model_dim, BLOCK_D):
                tl.store(
                    moe_buf_ptr + row * model_dim + d0 + offs_d,
                    tl.zeros((BLOCK_D,), dtype=moe_buf_ptr.dtype.element_ty),
                    mask=d0 + offs_d < model_dim,
                )
        offs_k = tl.arange(0, TOPK_PAD)
        if HAS_PAD_COUNT:
            if row >= n_valid:
                k_mask = offs_k < TOPK
                tl.store(
                    topk_ids_ptr + row * TOPK + offs_k,
                    tl.zeros((TOPK_PAD,), dtype=tl.int32),
                    mask=k_mask,
                )
                tl.store(
                    topk_weights_ptr + row * TOPK + offs_k,
                    tl.zeros((TOPK_PAD,), dtype=tl.float32),
                    mask=k_mask,
                )
        return

    t = tl.arange(0, M_PAD)
    k = tl.arange(0, TOPK_PAD)
    in_range = (t[:, None] < num_tokens) & (k[None, :] < TOPK)
    e = tl.load(topk_ids_ptr + t[:, None] * TOPK + k[None, :], mask=in_range, other=0)
    w = tl.load(
        topk_weights_ptr + t[:, None] * TOPK + k[None, :], mask=in_range, other=0.0
    )
    padded_row = (t >= n_valid)[:, None]
    e = tl.where(padded_row, 0, e)
    w = tl.where(padded_row, 0.0, w)
    loc = tl.load(local_expert_ids_ptr + e, mask=in_range, other=-1)
    loc = tl.where(in_range, loc, -1)  # [M_PAD, TOPK_PAD], -1 = not local

    # Opus keeps one entry per (token, expert): a later slot with the same expert wins
    later_same = (
        (loc[:, :, None] == loc[:, None, :])
        & (k[None, None, :] > k[None, :, None])
        & (loc[:, :, None] >= 0)
    )
    is_dup = tl.sum(later_same.to(tl.int32), axis=2) > 0
    loc = tl.where(is_dup, -1, loc)

    # every program needs the global expert prefix; non-local ids go to a dummy bin
    el = tl.arange(0, E_PAD)
    bins = tl.where(loc < 0, E_PAD - 1, loc)
    cnt = tl.histogram(tl.reshape(bins, (M_PAD * TOPK_PAD,)), E_PAD)
    cnt = tl.where(el == E_PAD - 1, 0, cnt)
    padded = ((cnt + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    start = tl.cumsum(padded, axis=0) - padded
    total = tl.sum(padded, axis=0)

    # This program's slice of local experts.
    ec = pid * EC + tl.arange(0, EC)
    sel = el[None, :] == ec[:, None]
    start_c = tl.sum(tl.where(sel, start[None, :], 0), axis=1)
    cnt_c = tl.sum(tl.where(sel, cnt[None, :], 0), axis=1)
    padded_c = ((cnt_c + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    kp1 = (k + 1)[None, :, None]
    hit = loc[:, :, None] == ec[None, None, :]  # [M_PAD, TOPK_PAD, EC]
    mesh = tl.max(tl.where(hit, kp1, 0), axis=1)  # [M_PAD, EC]: winning slot + 1
    present = mesh > 0
    pres_i = present.to(tl.int32)

    if pid == 0:
        tl.store(num_valid_ids_ptr, total)
        tl.store(num_valid_ids_ptr + 1, num_tokens)

    # Tokens ascend inside an expert: rank = earlier tokens on the same expert.
    rank = tl.cumsum(pres_i, axis=0) - pres_i
    pos = start_c[None, :] + rank
    w_sel = tl.sum(
        tl.where(hit & (kp1 == mesh[:, None, :]), w[:, :, None], 0.0), axis=1
    )
    packed = ((mesh - 1) << 24) | t[:, None]
    tl.store(sorted_ids_ptr + pos, packed, mask=present)
    tl.store(sorted_weights_ptr + pos, w_sel, mask=present)

    # Block padding after each expert's entries.
    r = tl.arange(0, BLOCK_M)
    pad_pos = start_c[:, None] + cnt_c[:, None] + r[None, :]
    pad_mask = r[None, :] < (padded_c - cnt_c)[:, None]
    tl.store(
        sorted_ids_ptr + pad_pos,
        tl.full(pad_pos.shape, (TOPK << 24) | num_tokens, tl.int32),
        mask=pad_mask,
    )
    tl.store(
        sorted_weights_ptr + pad_pos, tl.zeros(pad_pos.shape, tl.float32), mask=pad_mask
    )

    # One local expert id per block.
    b = tl.arange(0, NB_PER_E)
    blk = start_c[:, None] // BLOCK_M + b[None, :]
    blk_mask = b[None, :] < (padded_c // BLOCK_M)[:, None]
    tl.store(
        sorted_expert_ids_ptr + blk,
        tl.broadcast_to(ec[:, None], blk.shape),
        mask=blk_mask,
    )

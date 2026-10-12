# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 context-parallel query plan: the movement, the frames, and their invariants.

The layout is one packed row's coordinate change, so every property worth pinning is
checkable without a device and without a process group: which rows a rank ends up holding,
that the movement is a permutation of the slab, that the receive side's closed form agrees
with the row-by-row derivation, and that the frames say the queries are the chunks while the
KV streams stay the row.

The collectives are emulated in process, from the plan's own tables -- the same tables the
forward hands to ``all_to_all_single`` -- because a plan is what this file is about; the
module's own call into the collective is a thin wrapper over them.
"""

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.context_parallel import (
    ExchangeMetadata,
    QueryExchange,
    cp_plan,
    document_bounds,
)
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata, DeepSeekV41Model

# Two ratios keep the alignment arithmetic small while still exercising the lcm.
_RATIOS = (0, 2, 1)


def _positions(*documents: int) -> torch.Tensor:
    """One packed row whose documents have the given lengths, positions resetting."""
    flat: list[int] = []
    for length in documents:
        flat += list(range(length))
    return torch.tensor([flat])


def _bounds(cu: list[int]) -> torch.Tensor:
    return torch.tensor(cu, dtype=torch.int32)


def _oracle_splits(cu: list[int], cp_size: int, seq_len: int) -> torch.Tensor:
    """``M[source][dest]`` by counting every slab row's destination, the slow obvious way."""
    columns = []
    bounds = _bounds(cu)
    chunk = (bounds[1:] - bounds[:-1]) // cp_size
    for source in range(cp_size):
        rows = torch.arange(source * seq_len, (source + 1) * seq_len)
        document = torch.bucketize(rows, bounds, right=True) - 1
        columns.append(torch.bincount((rows - bounds[document]) // chunk[document], minlength=cp_size))
    return torch.stack(columns)


def _oracle_held_rows(cu: list[int], cp_size: int, rank: int) -> list[int]:
    """Rank ``rank``'s rows: chunk ``rank`` of every document, in document order."""
    rows: list[int] = []
    for start, end in zip(cu[:-1], cu[1:]):
        chunk = (end - start) // cp_size
        rows += range(start + rank * chunk, start + (rank + 1) * chunk)
    return rows


def _oracle_positions(cu: list[int], cp_size: int, rank: int) -> list[int]:
    bounds = _bounds(cu)
    held = torch.tensor(_oracle_held_rows(cu, cp_size, rank))
    return (held - bounds[torch.bucketize(held, bounds, right=True) - 1]).tolist()


_CASES = [
    ((0, 8, 24), 2),
    ((0, 16, 48), 4),
    ((0, 8, 24, 48), 2),
    ((0, 64), 8),
    ((0, 8, 32, 40), 4),
    ((0, 16, 48, 80), 8),
    ((0, 32, 64, 96), 4),
    ((0, 8, 16, 24, 32), 4),
    ((0, 4096, 8192, 12288), 8),
]


def test_the_bounds_are_read_off_the_row():
    """The boundaries come from the position resets, which is where every frame starts."""
    for cu, _ in _CASES:
        assert document_bounds(_positions(*[e - s for s, e in zip(cu[:-1], cu[1:])])).tolist() == list(cu)


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_plan_matches_the_row(cu, cp_size):
    """Both split vectors and the positions, against a derivation that counts every row.

    The receive side is the one worth arguing about: it is a closed form over the documents'
    chunk intervals rather than a histogram of the rows this rank receives, so it is checked
    against the row-by-row matrix and not against itself.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    matrix = _oracle_splits(cu, cp_size, seq_len)
    assert matrix.sum().item() == cu[-1], "the splits must partition the row"
    for rank in range(cp_size):
        plan = cp_plan(_bounds(cu), cp_size, rank, seq_len)
        assert list(plan.send_split) == matrix[rank].tolist()
        assert list(plan.recv_split) == matrix[:, rank].tolist()
        assert plan.positions_q.tolist() == _oracle_positions(cu, cp_size, rank)


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_movement_is_a_permutation_of_the_slab(cu, cp_size):
    """Every rank presents one slab and receives one slab, addressed row by row.

    ``send_index`` being a permutation is what makes the reverse movement the inverse rather
    than the adjoint (nothing is copied, so nothing accumulates), and it is also what makes
    the receiver's buffer the held rows' own order -- which is why there is no scatter index.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    chunk = (torch.tensor(cu[1:]) - torch.tensor(cu[:-1])) // cp_size
    for rank in range(cp_size):
        plan = cp_plan(_bounds(cu), cp_size, rank, seq_len)
        assert sorted(plan.send_index.tolist()) == list(range(seq_len))
        assert sum(plan.send_split) == seq_len
        assert sum(plan.recv_split) == seq_len
        # A document's chunk is one interval and spans at most two slabs; the closed form
        # leans on that, so the property is pinned rather than assumed.
        start = torch.tensor(cu[:-1]) + rank * chunk
        span = (start + chunk - 1) // seq_len - start // seq_len + 1
        assert int(span.max()) <= 2


@pytest.mark.parametrize("cu, cp_size, why", [
    ((0, 24, 32), 16, "a length that is not a multiple of cp_size"),
    ((0, 22, 32), 4, "a length that is not a multiple of cp_size"),
    ((0, 32, 32, 32), 8, "a zero-length document"),
])
def test_the_plan_rejects_a_row_it_cannot_grid(cu, cp_size, why):
    """A row the launcher should never have produced fails here rather than downstream.

    ``// cp_size`` on a ragged length is silent, and a zero-length chunk turns the receive
    side's interval arithmetic into an out-of-range index, so both are checked up front.
    """
    with pytest.raises(ValueError, match="positive multiple of cp_size"):
        cp_plan(_bounds(list(cu)), cp_size, 0, max(1, list(cu)[-1] // cp_size))


@pytest.mark.parametrize(
    "documents, cp_size",
    [((48, 20), 4), ((16, 12), 4), ((32, 24), 8)],
    ids=[
        "20 is not a multiple of cp_size * lcm(2, 1) = 8",
        "12 is not a multiple of 8",
        "24 is not a multiple of cp_size * 2 = 16",
    ],
)
def test_the_sharded_path_checks_the_launcher_contract(documents, cp_size):
    """A row the launcher should never have produced fails here, per document.

    ``cp_size * lcm(ratios > 1)`` is the alignment the launcher pads to; without it a chunk is
    ragged (rows land in the wrong place) and a compressed prefix is a partial group (the kernels
    grid an axis the pooling did not produce).  Both failures are silent downstream, so the check
    is a precondition rather than a diagnostic.
    """
    positions = _positions(*documents)
    owner = _HookOwner()
    with pytest.raises(ValueError, match="document alignment"):
        owner.masks(positions, cp_size, cp_size - 1)


def test_the_exchange_carries_the_movement_and_the_positions():
    """The container holds nothing a consumer could work out for itself."""
    plan = cp_plan(_bounds([0, 8, 24]), 2, 1, 12)
    assert isinstance(plan, ExchangeMetadata)
    assert set(plan.__slots__) == {"send_index", "send_split", "recv_split", "positions_q"}
    assert plan.send_index.dtype == torch.int64 and plan.positions_q.dtype == torch.int64


def test_the_exchange_is_the_identity_without_a_plan():
    """No plan means no movement, so the same call serves the unsharded forward.

    The early return is what keeps the call sites branch-free and lets ``cp = 1`` run without a
    process group at all; an unwired exchange with a plan is still an error rather than a silent
    fall back onto the default group.
    """
    exchange = QueryExchange(QueryExchange.Config())
    tensor = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    assert exchange.group is None
    assert exchange.permute(tensor, None) is tensor
    assert exchange.unpermute(tensor, None) is tensor
    with pytest.raises(RuntimeError, match="parallelize"):
        exchange.permute(tensor, cp_plan(_bounds([0, 8, 24]), 2, 0, 12))


@pytest.mark.parametrize("cu, cp_size", _CASES)
def test_the_movement_round_trips_the_whole_row(cu, cp_size):
    """Moving the query side out and back is the identity, row for row.

    Emulated in process: each rank presents ``slab[send_index]``, the slices are routed to
    their peer's inbox in peer order, the receiver takes the concatenation as its own order
    (no scatter index), and the reverse swaps the split vectors and scatters through
    ``send_index``.  Every rank's present is taken from **its own** slab, which is the
    property that makes the tables usable at all.
    """
    cu = list(cu)
    seq_len = cu[-1] // cp_size
    plans = [cp_plan(_bounds(cu), cp_size, rank, seq_len) for rank in range(cp_size)]
    slabs = [torch.arange(r * seq_len, (r + 1) * seq_len, dtype=torch.float32).reshape(-1, 1) for r in range(cp_size)]
    held = _emulate(slabs, plans)
    for rank in range(cp_size):
        assert held[rank].reshape(-1).tolist() == [float(row) for row in _oracle_held_rows(cu, cp_size, rank)]
    back = _emulate(held, plans, inverse=True)
    for rank in range(cp_size):
        torch.testing.assert_close(back[rank], slabs[rank])


def _emulate(tensors: list[torch.Tensor], plans: list[ExchangeMetadata], *, inverse: bool = False):
    """Run the pair of collectives over every rank, in process."""
    size = len(tensors)
    presents = [t if inverse else t[plans[r].send_index] for r, t in enumerate(tensors)]
    splits = [
        (plans[r].recv_split, plans[r].send_split) if inverse else (plans[r].send_split, plans[r].recv_split)
        for r in range(size)
    ]
    offsets = []
    for send, _ in splits:
        running, rows = [0], 0
        for value in send:
            rows += value
            running.append(rows)
        offsets.append(running)
    outs = []
    for rank in range(size):
        received = torch.cat(
            [
                presents[source][offsets[source][rank] : offsets[source][rank + 1]]
                for source in range(size)
            ]
        )
        if inverse:
            slab = torch.empty((sum(splits[rank][0]), *received.shape[1:]), dtype=received.dtype)
            slab[plans[rank].send_index] = received
            outs.append(slab)
        else:
            outs.append(received)
    return outs


def test_the_frames_say_the_queries_are_the_chunks_and_the_kv_streams_are_the_row():
    """The three frames under CP: chunk queries, row KV streams, and the reach in ``seqused``.

    ``seqused`` is the whole point of this layout on the kernel side: the KV tensors are the
    gathered row -- so their ``cu_seqlens`` must address that row -- while the queries only
    reach ``K_d = (cp_rank + 1) · Q_d`` of each document, which is what the kernels' end-aligned
    window and causal limit are anchored at.
    """
    for cu, cp_size in _CASES:
        chunk = (torch.tensor(cu[1:]) - torch.tensor(cu[:-1])) // cp_size
        for rank in range(cp_size):
            masks = _HookOwner().masks(_positions(*[e - s for s, e in zip(cu[:-1], cu[1:])]), cp_size, rank)
            assert masks.exchange is not None
            assert masks.q.cu_seqlens.tolist() == (torch.tensor(cu) // cp_size).tolist()
            assert masks.q.seqused.tolist() == chunk.tolist()
            assert masks.swa_k.cu_seqlens.tolist() == list(cu)
            assert masks.swa_k.seqused.tolist() == ((rank + 1) * chunk).tolist()
            for ratio, frame in masks.cmp_k.items():
                assert frame.cu_seqlens.tolist() == (torch.tensor(cu) // ratio).tolist()
                assert frame.seqused.tolist() == (((rank + 1) * chunk) // ratio).tolist()
                if ratio == 1:
                    assert frame.residual is None
                else:
                    assert frame.residual.tolist() == [0] * (len(cu) - 1)
            # ``positions_q`` rides shaped like the input, so the oracle is compared flat.
            assert masks.positions_q.reshape(-1).tolist() == _oracle_positions(list(cu), cp_size, rank)


def test_the_unsharded_forward_is_unchanged():
    """CP=1 keeps every frame the row, for every stack.

    The grid's arithmetic degenerates on its own at ``cp_size == 1`` -- one chunk per document
    *is* the document -- so the unsharded path is the same derivation rather than a branch, and
    the metadata it builds is the one geometry both the torch-native and the fused paths read.
    """
    masks = _HookOwner().masks(_positions(8, 16), 1, 0)
    assert masks.exchange is None
    torch.testing.assert_close(masks.positions_q, _positions(8, 16))
    assert masks.q.cu_seqlens.tolist() == [0, 8, 24]
    assert masks.q.seqused.tolist() == [8, 16]
    assert masks.swa_k.cu_seqlens.tolist() == [0, 8, 24]
    assert masks.swa_k.seqused.tolist() == [8, 16]
    assert sorted(masks.cmp_k) == [1, 2]
    assert masks.cmp_k[1].seqused.tolist() == [8, 16]
    assert masks.cmp_k[2].seqused.tolist() == [4, 8]
    assert masks.cmp_k[2].cu_seqlens.tolist() == [0, 4, 12]


def test_the_hook_attaches_the_plan_and_slices_the_inputs():
    """``build_attention_masks`` derives the plan from the **global** row, then shards.

    The plan cannot be derived from the slab, and the frames cannot either, so this is the one
    place both the whole row and the mesh meet: the inputs it returns are the slab in plain
    rank order, the positions it forwards describe that slab, and the metadata carries the
    plan those rows were addressed by.
    """
    row = _positions(8, 16)
    tokens = torch.zeros_like(row)
    for cp_size in (2, 4):
        for cp_rank in range(cp_size):
            extra = {"positions": row.clone()}
            inputs, _, extra = _HookOwner().build_attention_masks(
                tokens, tokens, extra, cp_mesh=_Mesh(cp_size, cp_rank)
            )
            slab = row.numel() // cp_size
            torch.testing.assert_close(extra["positions"], row[..., cp_rank * slab : (cp_rank + 1) * slab])
            assert inputs.shape[-1] == slab
            masks = extra["attention_masks"]
            assert masks.exchange is not None
            plan = masks.exchange
            assert list(plan.send_split) == _oracle_splits([0, 8, 24], cp_size, slab)[cp_rank].tolist()
            assert sum(plan.recv_split) == slab
            assert sorted(plan.send_index.tolist()) == list(range(slab))


def test_the_model_takes_a_context_parallel_degree():
    """The text stack admits CP; the layout above is what it was waiting for."""
    assert DeepSeekV41Model.Config.accepts_context_parallel is True


class _HookOwner:
    """The model's metadata builders, without the rest of the model.

    ``_frame`` and the hook have to stay staticmethods/methods when they are grafted on, or
    ``self`` becomes their first argument.
    """

    compress_ratios = _RATIOS
    engram_history_size = 0
    # The model's mask post-process seam: a stand-in bypasses ``__init__``, so it
    # states the pre-extension answer -- none, the masks pass through untouched.
    _metadata_extension = None
    # The hook is a method, so a stand-in presents the pieces it reaches for: the frame builders,
    # taken from the model itself, and the sharded half.  ``_frame`` is a staticmethod there, so it
    # is grafted as one -- a plain assignment would make it an instance method here.
    get_attention_masks = DeepSeekV41Model.get_attention_masks
    _row_metadata = DeepSeekV41Model._row_metadata
    _sharded_attention_masks = DeepSeekV41Model._sharded_attention_masks
    build_attention_masks = DeepSeekV41Model.build_attention_masks
    _frame = staticmethod(DeepSeekV41Model._frame)

    def masks(self, positions, cp_size: int = 1, cp_rank: int = 0) -> DeepSeekV41Metadata:
        """The hook's two paths, without the slicing: the row's metadata, or the sharded one."""
        if cp_size == 1:
            return self.get_attention_masks(positions)
        slab = positions.numel() // cp_size
        lo = cp_rank * slab
        return self._sharded_attention_masks(
            self._row_metadata(positions),
            positions[..., lo : lo + slab],
            cp_size=cp_size,
            cp_rank=cp_rank,
        )


class _Mesh:
    """The two ``cp_mesh`` methods the hook reads, without a real process group."""

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank

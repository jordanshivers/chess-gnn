import io
from pathlib import Path

import chess
import torch
from torch_geometric.data import Batch

from chess_gnn.dataset import PGNMoveDataset

TINY_PGN = """[Event "t"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1/2-1/2

[Event "t2"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
"""


def test_dataset_round_trip(tmp_path: Path):
    p = tmp_path / "tiny.pgn"
    p.write_text(TINY_PGN)
    ds = PGNMoveDataset([p], shuffle_buffer=4, min_elo=2200)
    items = list(iter(ds))
    assert len(items) == 10 + 6  # plies across both games
    for d in items:
        assert d.x.shape[0] == 64
        assert d.y.dtype == torch.long
        assert 0 <= int(d.y.item()) < 4096
        assert d.legal_mask[int(d.y.item())].item() is True
        assert d.value_target.dtype == torch.float32
    batch = Batch.from_data_list(items[:4])
    assert batch.y.shape == (4,)
    assert batch.legal_mask.view(4, 4096).shape == (4, 4096)
    assert batch.value_target.shape == (4,)


def test_dataset_value_target_is_from_side_to_move(tmp_path: Path):
    p = tmp_path / "tiny.pgn"
    p.write_text(TINY_PGN)
    ds = PGNMoveDataset([p], shuffle_buffer=64, min_elo=2200)
    values = sorted(float(d.value_target.item()) for d in ds)

    assert values.count(-1.0) == 3
    assert values.count(0.0) == 10
    assert values.count(1.0) == 3

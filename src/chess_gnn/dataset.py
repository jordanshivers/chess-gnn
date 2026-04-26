"""Streaming PGN dataset of (position, move) training pairs.

Supports plain `.pgn` files and zstd-compressed `.pgn.zst` dumps (Lichess format).
The dataset is an `IterableDataset` so we never materialize the full corpus in
memory; a small shuffle buffer gives training locality without breaking streaming.
"""

from __future__ import annotations

import io
import random
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch_geometric.data import Data

from .encoding import board_to_data
from .moves import is_promoting_pawn_move, move_to_index

# Mapping from promotion piece -> under-promotion target index (0..2), or -1 for queen/none.
_UNDERPROMO_INDEX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def _result_to_white_value(result: str) -> float:
    return {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)


def _open_pgn_stream(path: Path) -> IO[str]:
    if path.suffix == ".zst":
        import zstandard as zstd  # type: ignore

        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    with _open_pgn_stream(path) as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game


def _iter_positions(
    path: Path,
    min_elo: int | None = None,
) -> Iterator[tuple[chess.Board, chess.Move, float]]:
    """Yield (board-before-move, move, white-POV result) tuples from PGNs."""
    for game in _iter_games(path):
        if min_elo is not None:
            try:
                we = int(game.headers.get("WhiteElo", "0"))
                be = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                continue
            if min(we, be) < min_elo:
                continue
        board = game.board()
        white_value = _result_to_white_value(game.headers.get("Result", "*"))
        for move in game.mainline_moves():
            yield board.copy(stack=False), move, white_value
            board.push(move)


class PGNMoveDataset(IterableDataset):
    """Streaming (position, move) dataset over one or more PGN files.

    Each iteration yields a PyG `Data` object with extra fields:
      - `y`: the played-move index in [0, 4096)
      - `underpromo_target`: int in {-1, 0, 1, 2} (-1 = not an under-promotion)
      - `legal_mask`: bool tensor [4096]
      - `value_target`: game result from the side-to-move's perspective
    """

    def __init__(
        self,
        paths: list[Path] | list[str],
        shuffle_buffer: int = 8192,
        min_elo: int | None = 2200,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.shuffle_buffer = shuffle_buffer
        self.min_elo = min_elo
        self.seed = seed

    def _worker_paths(self) -> list[Path]:
        info = get_worker_info()
        if info is None:
            return list(self.paths)
        # Shard files across workers, round-robin.
        return [p for i, p in enumerate(self.paths) if i % info.num_workers == info.id]

    def _raw_stream(self) -> Iterator[tuple[chess.Board, chess.Move, float]]:
        for path in self._worker_paths():
            yield from _iter_positions(path, min_elo=self.min_elo)

    def __iter__(self) -> Iterator[Data]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + worker_id)

        buf: list[tuple[chess.Board, chess.Move, float]] = []
        raw = self._raw_stream()

        # Fill the buffer
        for item in raw:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                break

        for item in raw:
            swap_idx = rng.randrange(len(buf))
            out_board, out_move, out_value = buf[swap_idx]
            buf[swap_idx] = item
            yield self._encode(out_board, out_move, out_value)

        rng.shuffle(buf)
        for out_board, out_move, out_value in buf:
            yield self._encode(out_board, out_move, out_value)

    @staticmethod
    def _encode(board: chess.Board, move: chess.Move, white_value: float) -> Data:
        from .moves import legal_mask

        data = board_to_data(board)
        data.y = torch.tensor(move_to_index(move), dtype=torch.long)
        if is_promoting_pawn_move(board, move) and move.promotion in _UNDERPROMO_INDEX:
            up = _UNDERPROMO_INDEX[move.promotion]
        else:
            up = -1
        data.underpromo_target = torch.tensor(up, dtype=torch.long)
        data.legal_mask = legal_mask(board)
        side_value = white_value if board.turn == chess.WHITE else -white_value
        data.value_target = torch.tensor(side_value, dtype=torch.float32)
        return data


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class EngineValueDataset(Dataset):
    """Finite dataset of Stockfish-labeled PGN positions.

    The file is produced by `python -m chess_gnn.make_value_dataset` and stores
    primitive records containing FEN, played move, and a side-to-move engine
    value in [-1, 1].
    """

    def __init__(self, path: Path | str, value_blend: float = 1.0) -> None:
        self.path = Path(path)
        payload = _torch_load(self.path)
        self.records = payload["records"] if isinstance(payload, dict) else payload
        if not self.records:
            raise ValueError(f"No value records found in {self.path}")
        self.value_blend = float(value_blend)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        record = self.records[idx]
        board = chess.Board(record["fen"])
        move = chess.Move.from_uci(record["move_uci"])
        data = PGNMoveDataset._encode(board, move, float(record.get("game_value", 0.0)))
        engine_value = float(record["engine_value"])
        game_value = float(record.get("game_value", 0.0))
        value = self.value_blend * engine_value + (1.0 - self.value_blend) * game_value
        data.value_target = torch.tensor(value, dtype=torch.float32)
        data.engine_value_target = torch.tensor(engine_value, dtype=torch.float32)
        data.game_value_target = torch.tensor(game_value, dtype=torch.float32)
        return data

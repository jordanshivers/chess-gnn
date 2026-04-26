"""Generate Stockfish value labels for sampled PGN positions.

The output is a torch-serialized dict containing primitive Python records:
```
{
    "metadata": {...},
    "records": [
        {
            "fen": "...",
            "move_uci": "e2e4",
            "engine_value": 0.23,  # side-to-move POV, in [-1, 1]
            "game_value": 1.0,     # side-to-move final-result target
            ...
        },
    ],
}
```
These records are intentionally simple so they remain easy to inspect and load.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
import chess.pgn
import torch
from tqdm import tqdm

from .dataset import _open_pgn_stream, _result_to_white_value


def _iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    with _open_pgn_stream(path) as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game


def _passes_elo(game: chess.pgn.Game, min_elo: int | None) -> bool:
    if min_elo is None:
        return True
    try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
    except ValueError:
        return False
    return min(white_elo, black_elo) >= min_elo


def _score_to_value(score: chess.engine.PovScore, turn: chess.Color, cp_scale: float) -> float:
    cp = score.pov(turn).score(mate_score=100_000)
    if cp is None:
        return 0.0
    return float(math.tanh(cp / cp_scale))


def _save(path: Path, metadata: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "records": records}, path)


def generate(
    pgn_paths: list[Path],
    out: Path,
    stockfish: str,
    positions: int = 200_000,
    depth: int | None = 8,
    move_time: float | None = None,
    min_elo: int | None = 1800,
    min_ply: int = 8,
    sample_every: int = 4,
    max_positions_per_game: int = 8,
    cp_scale: float = 400.0,
    seed: int = 0,
    save_every: int = 1000,
) -> None:
    if not shutil.which(stockfish) and not Path(stockfish).exists():
        raise FileNotFoundError(f"Stockfish binary not found: {stockfish}")
    if depth is None and move_time is None:
        raise ValueError("Pass either depth or move_time.")

    rng = random.Random(seed)
    metadata = {
        "pgn_paths": [str(p) for p in pgn_paths],
        "stockfish": stockfish,
        "positions": positions,
        "depth": depth,
        "move_time": move_time,
        "min_elo": min_elo,
        "min_ply": min_ply,
        "sample_every": sample_every,
        "max_positions_per_game": max_positions_per_game,
        "cp_scale": cp_scale,
        "seed": seed,
    }
    records: list[dict] = []

    limit = chess.engine.Limit(depth=depth) if move_time is None else chess.engine.Limit(time=move_time)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    try:
        progress = tqdm(total=positions, desc="stockfish labels")
        for pgn_path in pgn_paths:
            for game in _iter_games(pgn_path):
                if len(records) >= positions:
                    break
                if not _passes_elo(game, min_elo):
                    continue

                moves = list(game.mainline_moves())
                candidate_plies = [
                    i for i in range(len(moves))
                    if i >= min_ply and (i - min_ply) % max(sample_every, 1) == 0
                ]
                rng.shuffle(candidate_plies)
                selected = set(candidate_plies[:max_positions_per_game])

                board = game.board()
                white_game_value = _result_to_white_value(game.headers.get("Result", "*"))
                for ply, move in enumerate(moves):
                    if len(records) >= positions:
                        break
                    if ply in selected:
                        info = engine.analyse(board, limit)
                        score = info.get("score")
                        if score is not None:
                            game_value = (
                                white_game_value if board.turn == chess.WHITE else -white_game_value
                            )
                            records.append(
                                {
                                    "fen": board.fen(),
                                    "move_uci": move.uci(),
                                    "engine_value": _score_to_value(score, board.turn, cp_scale),
                                    "game_value": game_value,
                                    "ply": ply,
                                    "result": game.headers.get("Result", "*"),
                                }
                            )
                            progress.update(1)
                            if save_every > 0 and len(records) % save_every == 0:
                                _save(out, metadata, records)
                    board.push(move)
            if len(records) >= positions:
                break
        progress.close()
    finally:
        engine.quit()

    _save(out, metadata, records)
    print(f"wrote {len(records)} records to {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--positions", type=int, default=200_000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--move-time", type=float, default=None)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--max-positions-per-game", type=int, default=8)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(
        pgn_paths=args.pgn,
        out=args.out,
        stockfish=args.stockfish,
        positions=args.positions,
        depth=args.depth,
        move_time=args.move_time,
        min_elo=args.min_elo,
        min_ply=args.min_ply,
        sample_every=args.sample_every,
        max_positions_per_game=args.max_positions_per_game,
        cp_scale=args.cp_scale,
        seed=args.seed,
        save_every=args.save_every,
    )

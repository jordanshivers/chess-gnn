"""Visualization helpers: render a board with predicted-move arrows."""

from __future__ import annotations

import chess
import chess.svg

from .play import MoveRanking


def _color_for_prob(p: float, base_hue: str = "#3366ff") -> str:
    """Map a probability in [0, 1] to an rgba-ish hex string with opacity."""
    alpha = int(40 + 215 * max(0.0, min(1.0, p)))  # 40..255
    return f"{base_hue}{alpha:02x}"


def render_prediction_svg(
    board: chess.Board,
    ranking: MoveRanking,
    topk: int = 8,
    size: int = 420,
) -> str:
    """Return an SVG string of the board with the top-k predicted moves drawn as arrows.

    Opacity encodes probability; top move is rendered brightest.
    """
    topk = min(topk, len(ranking.moves))
    if topk == 0:
        return chess.svg.board(board, size=size)

    max_p = max(ranking.probabilities[:topk]) or 1.0
    arrows = []
    for mv, p in zip(ranking.moves[:topk], ranking.probabilities[:topk]):
        color = _color_for_prob(p / max_p)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color))
    return chess.svg.board(board, arrows=arrows, size=size)


def square_heatmap(
    board: chess.Board,
    ranking: MoveRanking,
    from_square: int,
) -> dict[int, float]:
    """Aggregate the ranking's probability mass for moves originating at `from_square`.

    Returns a dict {to_square: probability}, useful for coloring destination squares
    in a per-piece heatmap view.
    """
    out: dict[int, float] = {}
    for mv, p in zip(ranking.moves, ranking.probabilities):
        if mv.from_square == from_square:
            out[mv.to_square] = out.get(mv.to_square, 0.0) + p
    return out

"""Move <-> index mapping and legal-move masking.

The main policy head produces one logit per directed (from, to) edge, flattened
to shape [4096]. Promotions default to queen; a separate 3-way under-promotion
head (N/B/R) is queried only on promoting pawn moves.
"""

from __future__ import annotations

import chess
import torch

NUM_MOVES = 64 * 64  # 4096
UNDERPROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def move_to_index(move: chess.Move) -> int:
    """Map a `chess.Move` to its (from, to) index in [0, 4096)."""
    return move.from_square * 64 + move.to_square


def index_to_from_to(index: int) -> tuple[int, int]:
    return divmod(index, 64)


def is_promoting_pawn_move(board: chess.Board, move: chess.Move) -> bool:
    """True if `move` is a pawn move to the last rank (i.e. requires promotion)."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(move.to_square)
    return (piece.color == chess.WHITE and to_rank == 7) or (
        piece.color == chess.BLACK and to_rank == 0
    )


def legal_mask(board: chess.Board) -> torch.Tensor:
    """Boolean mask over the 4096 (from, to) moves, True on legal moves.

    All four promotion choices share the same (from, to) index; we mark the
    index legal if *any* promotion (typically queen) is legal there.
    """
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv)] = True
    return mask


def decode_move(
    board: chess.Board,
    index: int,
    underpromo_choice: int | None = None,
) -> chess.Move:
    """Turn a policy index (and optional under-promo choice 0..2) into a legal move.

    `underpromo_choice` selects N/B/R (0/1/2). If None, promotions default to queen.
    Raises ValueError if the resulting move is not legal in `board`.
    """
    from_sq, to_sq = index_to_from_to(index)
    trial = chess.Move(from_sq, to_sq)
    if is_promoting_pawn_move(board, trial):
        promo_piece = (
            UNDERPROMO_PIECES[underpromo_choice]
            if underpromo_choice is not None
            else chess.QUEEN
        )
        move = chess.Move(from_sq, to_sq, promotion=promo_piece)
    else:
        move = trial
    if move not in board.legal_moves:
        raise ValueError(f"Decoded move {move.uci()} is not legal in position {board.fen()}")
    return move

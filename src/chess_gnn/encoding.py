"""Board <-> PyG Data encoding.

Each square is a node in a fully-connected directed graph (64*63 edges).
Node features describe the piece on that square plus position; global game
state (side to move, castling, en passant, halfmove clock) is concatenated
onto every node so message passing has access to it.

Edge features encode geometric priors (file/rank deltas, slider/knight
compatibility) so the network does not have to rediscover chess geometry.
"""

from __future__ import annotations

import chess
import numpy as np
import torch
from torch_geometric.data import Data

NODE_FEATURE_DIM = 45
EDGE_FEATURE_DIM = 7
NUM_SQUARES = 64

# Ordered piece list used for the per-square one-hot (index 0 = empty).
_PIECE_ORDER = [
    None,
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]
_PIECE_TO_INDEX = {p: i for i, p in enumerate(_PIECE_ORDER) if p is not None}


def _edge_index_and_attr() -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the fully-connected directed edge list and edge features."""
    edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
    for i in range(NUM_SQUARES):
        fi, ri = chess.square_file(i), chess.square_rank(i)
        for j in range(NUM_SQUARES):
            if i == j:
                continue
            fj, rj = chess.square_file(j), chess.square_rank(j)
            df, dr = fj - fi, rj - ri
            adf, adr = abs(df), abs(dr)
            same_rank = float(dr == 0)
            same_file = float(df == 0)
            same_diag = float(adf == adr)
            is_knight = float((adf, adr) in ((1, 2), (2, 1)))
            cheby = max(adf, adr)
            edges.append((i, j))
            attrs.append([df / 7.0, dr / 7.0, same_rank, same_file, same_diag, is_knight, cheby / 7.0])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


_EDGE_INDEX, _EDGE_ATTR = _edge_index_and_attr()


def _global_features(board: chess.Board) -> np.ndarray:
    feats = np.zeros(15, dtype=np.float32)
    feats[0] = 1.0 if board.turn == chess.WHITE else 0.0
    feats[1] = float(board.has_kingside_castling_rights(chess.WHITE))
    feats[2] = float(board.has_queenside_castling_rights(chess.WHITE))
    feats[3] = float(board.has_kingside_castling_rights(chess.BLACK))
    feats[4] = float(board.has_queenside_castling_rights(chess.BLACK))
    # en passant one-hot over files 0..7 plus a "none" slot at index 13
    if board.ep_square is not None:
        feats[5 + chess.square_file(board.ep_square)] = 1.0
    else:
        feats[13] = 1.0
    feats[14] = min(board.halfmove_clock, 100) / 100.0
    return feats


def board_to_data(board: chess.Board) -> Data:
    """Encode a `chess.Board` as a PyG `Data` object.

    Returns a graph with 64 nodes, 4032 directed edges, node feature dim 45,
    edge feature dim 7.
    """
    x = np.zeros((NUM_SQUARES, NODE_FEATURE_DIM), dtype=np.float32)
    g = _global_features(board)

    for sq in range(NUM_SQUARES):
        piece = board.piece_at(sq)
        piece_idx = _PIECE_TO_INDEX[(piece.piece_type, piece.color)] if piece is not None else 0
        x[sq, piece_idx] = 1.0
        # square color: light if (file + rank) is odd
        file_ = chess.square_file(sq)
        rank_ = chess.square_rank(sq)
        x[sq, 13] = float((file_ + rank_) % 2 == 1)
        x[sq, 14 + rank_] = 1.0           # rank one-hot (14..21)
        x[sq, 22 + file_] = 1.0           # file one-hot (22..29)
        x[sq, 30:45] = g                  # global features broadcast

    data = Data(
        x=torch.from_numpy(x),
        edge_index=_EDGE_INDEX,
        edge_attr=_EDGE_ATTR,
    )
    data.turn = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
    return data

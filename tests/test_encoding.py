import chess
import torch

from chess_gnn.encoding import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_SQUARES,
    board_to_data,
)


def test_starting_position_shapes():
    data = board_to_data(chess.Board())
    assert data.x.shape == (NUM_SQUARES, NODE_FEATURE_DIM)
    assert data.edge_index.shape == (2, NUM_SQUARES * (NUM_SQUARES - 1))
    assert data.edge_attr.shape == (NUM_SQUARES * (NUM_SQUARES - 1), EDGE_FEATURE_DIM)
    # exactly one piece one-hot per square, and all squares accounted for
    piece_onehot = data.x[:, :13]
    assert torch.allclose(piece_onehot.sum(dim=1), torch.ones(NUM_SQUARES))


def test_piece_placement_matches():
    board = chess.Board()
    data = board_to_data(board)
    # e1 should encode white king
    e1 = chess.E1
    assert data.x[e1, 6] == 1.0  # white king index (see _PIECE_ORDER)
    # e8 should encode black king
    e8 = chess.E8
    assert data.x[e8, 12] == 1.0
    # empty square e4
    e4 = chess.E4
    assert data.x[e4, 0] == 1.0


def test_turn_indicator_round_trip():
    board = chess.Board()
    board.push_san("e4")
    data = board_to_data(board)
    assert data.turn.item() == 0.0  # black to move

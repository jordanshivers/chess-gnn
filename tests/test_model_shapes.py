import chess
import torch
from torch_geometric.data import Batch

from chess_gnn.encoding import board_to_data
from chess_gnn.model import ChessGNN, masked_log_softmax
from chess_gnn.moves import NUM_MOVES, legal_mask


def test_forward_shapes_and_no_nan():
    boards = [chess.Board() for _ in range(4)]
    batch = Batch.from_data_list([board_to_data(b) for b in boards])
    model = ChessGNN(hidden_dim=32, num_layers=2, num_heads=2)
    model.eval()
    with torch.no_grad():
        policy, underpromo, value = model(batch)
    assert policy.shape == (4, NUM_MOVES)
    assert underpromo.shape == (4, NUM_MOVES, 3)
    assert value.shape == (4,)
    assert torch.isfinite(policy[policy != float("-inf")]).all()
    assert torch.isfinite(underpromo).all()
    assert torch.isfinite(value).all()


def test_masked_log_softmax_respects_legal_moves():
    board = chess.Board()
    data = board_to_data(board)
    batch = Batch.from_data_list([data])
    model = ChessGNN(hidden_dim=32, num_layers=2, num_heads=2)
    model.eval()
    with torch.no_grad():
        policy, _, _ = model(batch)
    mask = legal_mask(board).unsqueeze(0)
    log_probs = masked_log_softmax(policy, mask)
    probs = log_probs.exp()
    # probabilities sum to ~1 over legal moves, exactly 0 on illegal ones
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert probs[~mask].abs().max().item() == 0.0

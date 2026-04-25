import torch

from chess_gnn.model import ChessGNN
from chess_gnn.selfplay import play_self_game


def test_self_play_produces_valid_trajectory():
    model = ChessGNN(hidden_dim=32, num_layers=2, num_heads=2)
    traj = play_self_game(model, device="cpu", temperature=1.0, max_plies=30)

    assert len(traj.positions) == len(traj.actions) == len(traj.masks) == len(traj.rewards)
    assert len(traj.positions) > 0
    # Every recorded action should have been legal in its position.
    for action, mask in zip(traj.actions, traj.masks):
        assert mask[action].item() is True
    # Rewards are in {-1, 0, +1} (terminal-only).
    unique = set(traj.rewards)
    assert unique <= {-1.0, 0.0, 1.0}
    # Side signs alternate.
    for i in range(1, len(traj.side_signs)):
        assert traj.side_signs[i] == -traj.side_signs[i - 1]

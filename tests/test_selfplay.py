import torch

from chess_gnn.model import ChessGNN
from chess_gnn.selfplay import play_self_game, play_self_games_mcts_batched


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


def test_batched_mcts_self_play_produces_valid_trajectories():
    model = ChessGNN(hidden_dim=16, num_layers=1, num_heads=2)
    trajs = play_self_games_mcts_batched(
        model,
        num_games=2,
        device="cpu",
        num_simulations=2,
        max_plies=6,
    )

    assert len(trajs) == 2
    for traj in trajs:
        assert len(traj.positions) == len(traj.actions) == len(traj.masks)
        assert len(traj.policy_targets) == len(traj.positions)
        assert len(traj.positions) > 0
        for action, mask, target in zip(traj.actions, traj.masks, traj.policy_targets):
            assert mask[action].item() is True
            assert target.shape == (4096,)
            assert abs(float(target.sum().item()) - 1.0) < 1e-6

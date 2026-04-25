"""Sanity checks for the PUCT MCTS implementation."""
from __future__ import annotations

import chess
import pytest
import torch

from chess_gnn.mcts import MCTS
from chess_gnn.model import ChessGNN
from chess_gnn.play import GNNAgent


@pytest.fixture(scope="module")
def tiny_model() -> ChessGNN:
    torch.manual_seed(0)
    return ChessGNN(hidden_dim=16, num_layers=1, num_heads=2)


def test_mcts_returns_legal_move(tiny_model: ChessGNN) -> None:
    mcts = MCTS(tiny_model, device="cpu")
    board = chess.Board()
    move = mcts.search(board, num_simulations=8, temperature=0.0)
    assert move in board.legal_moves


def test_visit_distribution_sums_to_one(tiny_model: ChessGNN) -> None:
    mcts = MCTS(tiny_model, device="cpu")
    moves, probs = mcts.visit_distribution(chess.Board(), num_simulations=16)
    assert len(moves) == len(probs) > 0
    assert all(m in chess.Board().legal_moves for m in moves)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_visit_distribution_controls_root_noise(tiny_model: ChessGNN, monkeypatch) -> None:
    seen: list[bool] = []
    original = MCTS._expand

    def spy(self, node, board, add_dirichlet):
        seen.append(add_dirichlet)
        return original(self, node, board, add_dirichlet)

    monkeypatch.setattr(MCTS, "_expand", spy)
    board = chess.Board()

    MCTS(tiny_model, device="cpu").visit_distribution(board, num_simulations=0)
    MCTS(
        tiny_model,
        device="cpu",
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    ).visit_distribution(board, num_simulations=0)
    MCTS(
        tiny_model,
        device="cpu",
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    ).visit_distribution(board, num_simulations=0, add_dirichlet=False)

    assert seen == [False, True, False]


def test_batched_pending_eval_updates_visit_counts(tiny_model: ChessGNN) -> None:
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("d4")

    mcts_a = MCTS(tiny_model, device="cpu")
    mcts_b = MCTS(tiny_model, device="cpu")
    pending = [
        p for p in (
            mcts_a.run_simulation_round(board_a),
            mcts_b.run_simulation_round(board_b),
        )
        if p is not None
    ]

    mcts_a.evaluate_pending_batch(pending)
    moves_a, probs_a = mcts_a.root_visit_distribution(board_a)
    moves_b, probs_b = mcts_b.root_visit_distribution(board_b)

    assert moves_a and abs(sum(probs_a) - 1.0) < 1e-6
    assert moves_b and abs(sum(probs_b) - 1.0) < 1e-6
    assert all(m in board_a.legal_moves for m in moves_a)
    assert all(m in board_b.legal_moves for m in moves_b)


def test_advance_root_reuses_selected_child(tiny_model: ChessGNN) -> None:
    mcts = MCTS(tiny_model, device="cpu")
    board = chess.Board()
    pending = mcts.run_simulation_round(board)
    if pending is not None:
        mcts.evaluate_pending_batch([pending])

    move = mcts.select_root_move(board, temperature=0.0)
    action = move.from_square * 64 + move.to_square
    assert mcts.root is not None
    child = mcts.root.children[action]
    board.push(move)
    mcts.advance_root(move, board)

    assert mcts.root is child
    assert mcts.root_key == board.fen()


def test_agent_mcts_disabled_by_default(tiny_model: ChessGNN) -> None:
    agent = GNNAgent(tiny_model, device="cpu", default_temperature=0.0)
    # Monkey-patch the MCTS.search to make sure it isn't called in the default mode.
    called = {"n": 0}
    orig = agent._mcts.search
    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)
    agent._mcts.search = spy  # type: ignore[assignment]
    agent.select_move(chess.Board())
    assert called["n"] == 0


def test_agent_mcts_enabled(tiny_model: ChessGNN) -> None:
    agent = GNNAgent(
        tiny_model, device="cpu", default_temperature=0.0, num_simulations=4,
    )
    move = agent.select_move(chess.Board())
    assert move in chess.Board().legal_moves

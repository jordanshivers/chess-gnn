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

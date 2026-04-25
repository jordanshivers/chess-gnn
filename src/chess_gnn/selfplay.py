"""Self-play game generation for RL fine-tuning.

Two flavors:

- `play_self_game` — raw-policy sampling. Records the collecting policy's
  log-prob and value estimate at each step so PPO can compute a clipped ratio
  against them in the update phase. Fast, but at this skill level games mostly
  draw, so the terminal reward is 0 on most moves and pure policy-gradient
  methods see weak signal.
- `play_self_game_mcts` — AlphaZero-style. Each move runs PUCT MCTS and records
  the visit distribution as a dense per-move policy target. This gives signal
  on every move regardless of whether the game was decisive, which is the fix
  for the "no signal" problem when almost all self-play games draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import torch
from torch_geometric.data import Batch

from .encoding import board_to_data
from .mcts import MCTS
from .model import ChessGNN
from .moves import NUM_MOVES, decode_move, legal_mask


@dataclass
class SelfPlayTrajectory:
    # Graph encodings of every visited position, in order.
    positions: list = field(default_factory=list)          # list[Data]
    # Policy index of the chosen move at each step.
    actions: list[int] = field(default_factory=list)
    # Legal-move masks for each position, used to reconstruct the training loss.
    masks: list[torch.Tensor] = field(default_factory=list)
    # Per-step reward from the moving player's perspective (only terminal is nonzero).
    rewards: list[float] = field(default_factory=list)
    # +1 if white to move at step t, else -1. Lets us apply the final outcome sign-correctly.
    side_signs: list[int] = field(default_factory=list)
    # Optional MCTS visit-count distribution over the 4096 (from,to) move indices,
    # from the moving player's POV. Set by `play_self_game_mcts` only; empty for
    # raw-policy self-play. Used as a dense policy target for AlphaZero-style
    # distillation (nonzero per move regardless of game outcome).
    policy_targets: list[torch.Tensor] = field(default_factory=list)
    # Log-prob of the chosen action under the collecting policy (used by PPO
    # as the "old" policy in the clipped ratio). Populated by `play_self_game`.
    old_log_probs: list[float] = field(default_factory=list)
    # Value-head estimate at the collecting position, from the moving player's
    # POV. Used as a fixed baseline across PPO epochs.
    old_values: list[float] = field(default_factory=list)
    result: str = "*"


def _result_to_scalar(result: str) -> float:
    """Map PGN result string to a scalar from white's perspective."""
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result, 0.0)


@torch.no_grad()
def play_self_game(
    model: ChessGNN,
    device: torch.device | str = "cpu",
    temperature: float = 0.7,
    max_plies: int = 300,
) -> SelfPlayTrajectory:
    """Run one self-play game, recording positions/actions/rewards."""
    device = torch.device(device)
    model.eval()
    board = chess.Board()
    traj = SelfPlayTrajectory()

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        data = board_to_data(board)
        batch = Batch.from_data_list([data]).to(device)
        policy, _, value = model(batch)
        mask = legal_mask(board)
        # Sampling policy uses the training temperature. The stored old log-prob
        # is the sampling policy's log-prob of the chosen action — that's the
        # distribution we're doing importance sampling against in PPO.
        scaled = policy.squeeze(0).cpu() / max(temperature, 1e-6)
        scaled = scaled.masked_fill(~mask, torch.finfo(scaled.dtype).min)
        probs = torch.softmax(scaled, dim=-1)
        log_probs_all = torch.log_softmax(scaled, dim=-1)
        action = int(torch.multinomial(probs, 1).item())

        try:
            move = decode_move(board, action)
        except ValueError:
            # Fall back to the top legal move if the sampled one was a promotion-edge
            # edge case (shouldn't happen, but stay robust during early training).
            move = next(iter(board.legal_moves))
            action = move.from_square * 64 + move.to_square

        traj.positions.append(data)
        traj.actions.append(action)
        traj.masks.append(mask)
        traj.side_signs.append(1 if board.turn == chess.WHITE else -1)
        traj.rewards.append(0.0)
        traj.old_log_probs.append(float(log_probs_all[action].item()))
        traj.old_values.append(float(value.item()))
        board.push(move)

    traj.result = board.result(claim_draw=True)
    outcome = _result_to_scalar(traj.result)
    if traj.rewards:
        # Only terminal reward; apply per-side sign so each move's return matches
        # the perspective of the side that played it.
        for i in range(len(traj.rewards)):
            traj.rewards[i] = outcome * traj.side_signs[i]
    return traj


@torch.no_grad()
def play_self_game_mcts(
    model: ChessGNN,
    device: torch.device | str = "cpu",
    num_simulations: int = 64,
    temperature: float = 1.0,
    temperature_drop_ply: int = 30,
    max_plies: int = 300,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> SelfPlayTrajectory:
    """AlphaZero-style self-play: MCTS picks moves and its visit distribution
    is stored as the policy target for each position.

    Temperature schedule mirrors AlphaZero: `temperature` (usually 1.0) for the
    opening so early-game move choice has variety, then drop to ~0 after
    `temperature_drop_ply` so strong moves are played in the midgame/endgame.
    Dirichlet noise is added at the root during sim to encourage exploration.
    """
    dev = torch.device(device)
    model.eval()
    mcts = MCTS(
        model,
        device=dev,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
    )
    board = chess.Board()
    traj = SelfPlayTrajectory()

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        moves, probs = mcts.visit_distribution(
            board,
            num_simulations=num_simulations,
            add_dirichlet=True,
        )
        if not moves:
            break

        # Dense policy target over the 4096 (from,to) index space.
        target = torch.zeros(NUM_MOVES, dtype=torch.float32)
        for mv, p in zip(moves, probs):
            target[mv.from_square * 64 + mv.to_square] += float(p)

        t = temperature if ply < temperature_drop_ply else 0.0
        if t <= 1e-3:
            mv = moves[int(max(range(len(probs)), key=lambda i: probs[i]))]
        else:
            w = torch.tensor(probs, dtype=torch.float32).pow(1.0 / t)
            w = w / w.sum().clamp_min(1e-12)
            mv = moves[int(torch.multinomial(w, 1).item())]
        action = mv.from_square * 64 + mv.to_square

        traj.positions.append(board_to_data(board))
        traj.actions.append(action)
        traj.masks.append(legal_mask(board))
        traj.side_signs.append(1 if board.turn == chess.WHITE else -1)
        traj.rewards.append(0.0)
        traj.policy_targets.append(target)
        board.push(mv)

    traj.result = board.result(claim_draw=True)
    outcome = _result_to_scalar(traj.result)
    for i in range(len(traj.rewards)):
        traj.rewards[i] = outcome * traj.side_signs[i]
    return traj

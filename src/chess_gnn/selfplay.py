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


@dataclass
class _MCTSSelfPlayState:
    board: chess.Board
    traj: SelfPlayTrajectory
    mcts: MCTS
    ply: int = 0
    done: bool = False


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
    return play_self_games_mcts_batched(
        model,
        num_games=1,
        device=device,
        num_simulations=num_simulations,
        temperature=temperature,
        temperature_drop_ply=temperature_drop_ply,
        max_plies=max_plies,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
    )[0]


@torch.no_grad()
def play_self_games_mcts_batched(
    model: ChessGNN,
    num_games: int,
    device: torch.device | str = "cpu",
    num_simulations: int = 64,
    temperature: float = 1.0,
    temperature_drop_ply: int = 30,
    max_plies: int = 300,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> list[SelfPlayTrajectory]:
    """Run AlphaZero-style self-play games together, batching MCTS leaf evals.

    Each active game owns its own tree. During a simulation round we select one
    pending leaf per active game, evaluate all such leaves in a single model
    call, and back up into the corresponding trees.
    """
    if num_games <= 0:
        return []

    dev = torch.device(device)
    model.eval()
    states = [
        _MCTSSelfPlayState(
            board=chess.Board(),
            traj=SelfPlayTrajectory(),
            mcts=MCTS(
                model,
                device=dev,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
            ),
        )
        for _ in range(num_games)
    ]

    for state in states:
        state.mcts.prepare_root(state.board, add_dirichlet=False)

    while True:
        active = [
            s for s in states
            if not s.done
            and s.ply < max_plies
            and not s.board.is_game_over(claim_draw=True)
        ]
        if not active:
            break

        for state in active:
            state.mcts.prepare_root(state.board, add_dirichlet=False)
            state.mcts.add_root_dirichlet_noise()

        for _ in range(num_simulations):
            pending = []
            for state in active:
                leaf = state.mcts.run_simulation_round(state.board)
                if leaf is not None:
                    pending.append(leaf)
            if pending:
                active[0].mcts.evaluate_pending_batch(pending)

        for state in active:
            moves, probs = state.mcts.root_visit_distribution(state.board)
            if not moves:
                state.done = True
                continue

            target = torch.zeros(NUM_MOVES, dtype=torch.float32)
            for mv, p in zip(moves, probs):
                target[mv.from_square * 64 + mv.to_square] += float(p)

            t = temperature if state.ply < temperature_drop_ply else 0.0
            if t <= 1e-3:
                mv = moves[int(max(range(len(probs)), key=lambda i: probs[i]))]
            else:
                w = torch.tensor(probs, dtype=torch.float32).pow(1.0 / t)
                w = w / w.sum().clamp_min(1e-12)
                mv = moves[int(torch.multinomial(w, 1).item())]
            action = mv.from_square * 64 + mv.to_square

            state.traj.positions.append(board_to_data(state.board))
            state.traj.actions.append(action)
            state.traj.masks.append(legal_mask(state.board))
            state.traj.side_signs.append(1 if state.board.turn == chess.WHITE else -1)
            state.traj.rewards.append(0.0)
            state.traj.policy_targets.append(target)
            state.board.push(mv)
            state.ply += 1

            if state.board.is_game_over(claim_draw=True) or state.ply >= max_plies:
                state.done = True
            else:
                state.mcts.advance_root(mv, state.board)

    for state in states:
        state.traj.result = state.board.result(claim_draw=True)
        outcome = _result_to_scalar(state.traj.result)
        for i in range(len(state.traj.rewards)):
            state.traj.rewards[i] = outcome * state.traj.side_signs[i]
    return [state.traj for state in states]

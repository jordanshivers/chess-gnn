"""Estimate the Elo of a trained checkpoint by playing Stockfish at calibrated strengths.

Per opponent, plays `--games` games (colors alternated) and computes a point
estimate from the score plus a binomial 95% CI. If multiple opponents are
given, also fits a single pooled rating across them via maximum likelihood.

Fairness notes:
- No opening book, no time advantages — Stockfish is given a fixed move-time
  budget via --sf-move-time (default 0.1 s). Shorter times give weaker play even
  at a given UCI_Elo setting.
- The agent has no search — its Elo reflects raw pattern-recognition quality,
  which is typically a few hundred points below the same policy + MCTS.
- Evaluate at low temperature (default 0.05) so the agent plays near-argmax.

Usage:
    python -m chess_gnn.eval_elo \
        --ckpt checkpoints/sl/sl_final.pt \
        --stockfish /opt/homebrew/bin/stockfish \
        --opponents 1320 1500 1800 2100 \
        --games 40
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch

from .model import load_model
from .play import GNNAgent


@dataclass
class MatchResult:
    opponent_elo: int
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------

def score_to_elo_diff(score: float) -> float:
    """Convert expected score in (0, 1) to the rating gap (self - opponent)."""
    s = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def elo_diff_to_score(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def point_estimate(m: MatchResult) -> tuple[float, float]:
    """Return (estimated rating, 95% CI half-width) for a single match vs an opponent."""
    s = m.score
    r_hat = m.opponent_elo + score_to_elo_diff(s)
    # Clamp score away from 0/1 for the SE computation.
    s_c = min(max(s, 1e-3), 1 - 1e-3)
    # SE of mean score: sqrt(s(1-s)/N); treat draws as 0.5 outcomes.
    se_score = math.sqrt(s_c * (1 - s_c) / max(m.games, 1))
    # d(elo)/d(score) at score s: 400 / (ln(10) * s * (1-s))
    d_elo = 400.0 / (math.log(10.0) * s_c * (1 - s_c))
    ci = 1.96 * d_elo * se_score
    return r_hat, ci


def pooled_mle(results: list[MatchResult]) -> float:
    """Maximum-likelihood pooled rating across several opponents.

    Treats each game as Bernoulli with draws weighted as 0.5 wins + 0.5 losses
    (not exact, but a sensible aggregate for Elo fitting).
    """
    def score(rating: float) -> float:
        total = 0.0
        for m in results:
            e = elo_diff_to_score(rating - m.opponent_elo)
            s_i = m.score
            total += m.games * (s_i - e)  # derivative of log-lik w.r.t. R (up to constant)
        return total

    # Bisection over a wide plausible range.
    lo, hi = 0.0, 3500.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_match(
    agent: GNNAgent,
    stockfish_path: str,
    opponent_elo: int,
    games: int,
    sf_move_time: float,
    agent_temperature: float,
    max_plies: int,
) -> MatchResult:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(opponent_elo)})
    except chess.engine.EngineError as e:
        engine.quit()
        raise RuntimeError(
            f"Stockfish rejected UCI_Elo={opponent_elo}: {e}. "
            "Check your Stockfish's supported range."
        )

    wins = draws = losses = 0
    limit = chess.engine.Limit(time=sf_move_time)
    try:
        for i in range(games):
            board = chess.Board()
            agent_is_white = i % 2 == 0
            for _ in range(max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                if (board.turn == chess.WHITE) == agent_is_white:
                    move = agent.select_move(board, temperature=agent_temperature)
                else:
                    move = engine.play(board, limit).move
                    if move is None:
                        break
                board.push(move)

            result = board.result(claim_draw=True)
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == agent_is_white:
                wins += 1
            else:
                losses += 1

            if (i + 1) % max(1, games // 10) == 0 or i == games - 1:
                print(
                    f"  vs {opponent_elo:>4d} | {i+1:>3d}/{games}: "
                    f"W{wins} D{draws} L{losses} (score {(wins + 0.5*draws)/(i+1):.3f})"
                )
    finally:
        engine.quit()

    return MatchResult(opponent_elo=opponent_elo, games=games, wins=wins, draws=draws, losses=losses)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _load_agent(
    ckpt: Path,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    device: str,
    temperature: float,
    num_simulations: int = 0,
) -> GNNAgent:
    model = load_model(
        ckpt,
        device=device,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    return GNNAgent(
        model, device=device, default_temperature=temperature,
        num_simulations=num_simulations,
    )


def _resolve_stockfish(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("stockfish")
    if not found:
        raise SystemExit(
            "Stockfish binary not found. Install it (e.g. `brew install stockfish`) "
            "or pass --stockfish /path/to/stockfish."
        )
    return found


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--stockfish", default=None, help="Path to stockfish binary. Defaults to $PATH.")
    p.add_argument(
        "--opponents", type=int, nargs="+", default=[1320, 1600, 1900, 2200],
        help="Stockfish UCI_Elo targets to evaluate against.",
    )
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sf-move-time", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mcts-sims", type=int, default=0,
                   help="MCTS simulations per move (0 disables search).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sf_path = _resolve_stockfish(args.stockfish)
    agent = _load_agent(
        args.ckpt, args.hidden_dim, args.num_layers, args.num_heads, args.device,
        args.temperature, num_simulations=args.mcts_sims,
    )
    mode = f"MCTS ({args.mcts_sims} sims)" if args.mcts_sims > 0 else "raw policy"
    print(f"Stockfish: {sf_path} | games/opponent: {args.games} | sf move time: {args.sf_move_time}s")
    print(f"Agent: {args.ckpt} @ T={args.temperature} on {args.device} | {mode}")

    results: list[MatchResult] = []
    for elo in args.opponents:
        print(f"\n-- match vs Stockfish UCI_Elo={elo} --")
        m = play_match(
            agent=agent,
            stockfish_path=sf_path,
            opponent_elo=elo,
            games=args.games,
            sf_move_time=args.sf_move_time,
            agent_temperature=args.temperature,
            max_plies=args.max_plies,
        )
        results.append(m)

    print("\n=== per-opponent estimates ===")
    for m in results:
        r_hat, ci = point_estimate(m)
        print(
            f"  vs {m.opponent_elo:>4d}: W{m.wins} D{m.draws} L{m.losses} "
            f"(score {m.score:.3f}) -> Elo {r_hat:.0f} +/- {ci:.0f}"
        )

    if len(results) > 1:
        pooled = pooled_mle(results)
        # Rough pooled CI: aggregate SE via delta method.
        var = 0.0
        for m in results:
            e = elo_diff_to_score(pooled - m.opponent_elo)
            var += m.games * e * (1 - e)
        se_pooled = (400.0 / math.log(10.0)) / math.sqrt(max(var, 1e-6))
        ci = 1.96 * se_pooled
        print(f"\n=== pooled MLE rating ===")
        print(f"  Elo {pooled:.0f} +/- {ci:.0f} (95% CI)")


if __name__ == "__main__":
    main()

"""RL fine-tuning. Two modes:

- `algo="az"` (default): AlphaZero-style. Self-play uses PUCT MCTS and stores
  the visit distribution as a per-move policy target. The loss is
  cross-entropy to that distribution + MSE to the final outcome. Every move
  contributes signal, even in drawn games.
- `algo="ppo"`: PPO with a clipped-ratio surrogate and the value head as a
  baseline. Self-play records the collecting policy's log-prob and value at
  each step; the update phase runs multiple epochs on the same batch, clipping
  the importance ratio to `[1-clip, 1+clip]` to keep updates on-policy. This
  replaces the older REINFORCE path — it reuses each rollout much more
  efficiently and is far more stable under a noisy terminal-only reward.

Periodically plays evaluation games against a frozen baseline (the SL starting
point) and prints win/draw/loss counts as a regression guardrail.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .model import ChessGNN, masked_log_softmax
from .play import GNNAgent, play_game
from .selfplay import SelfPlayTrajectory, play_self_game, play_self_game_mcts


def _flatten_trajectories(
    trajs: list[SelfPlayTrajectory],
) -> dict:
    positions = []
    actions: list[int] = []
    returns: list[float] = []
    masks: list[torch.Tensor] = []
    pi_targets: list[torch.Tensor] = []
    old_logps: list[float] = []
    old_vals: list[float] = []
    has_pi = all(len(t.policy_targets) == len(t.positions) and t.policy_targets for t in trajs)
    has_old = all(len(t.old_log_probs) == len(t.positions) and t.old_log_probs for t in trajs)
    for t in trajs:
        positions.extend(t.positions)
        actions.extend(t.actions)
        returns.extend(t.rewards)  # terminal-only, already signed per mover
        masks.extend(t.masks)
        if has_pi:
            pi_targets.extend(t.policy_targets)
        if has_old:
            old_logps.extend(t.old_log_probs)
            old_vals.extend(t.old_values)
    if not positions:
        raise ValueError("No positions collected from self-play.")
    return {
        "batch": Batch.from_data_list(positions),
        "actions": torch.tensor(actions, dtype=torch.long),
        "returns": torch.tensor(returns, dtype=torch.float32),
        "masks": torch.stack(masks, dim=0),
        "pi_targets": torch.stack(pi_targets, dim=0) if has_pi else None,
        "old_log_probs": torch.tensor(old_logps, dtype=torch.float32) if has_old else None,
        "old_values": torch.tensor(old_vals, dtype=torch.float32) if has_old else None,
    }


def evaluate(
    model: ChessGNN,
    baseline: ChessGNN,
    device: str,
    num_games: int = 10,
    temperature: float = 0.2,
) -> dict:
    """Play `num_games` games vs the frozen baseline, alternating colors."""
    agent = GNNAgent(model, device=device, default_temperature=temperature)
    base_agent = GNNAgent(baseline, device=device, default_temperature=temperature)
    wins = draws = losses = 0
    for i in range(num_games):
        white, black = (agent, base_agent) if i % 2 == 0 else (base_agent, agent)
        _, result = play_game(white, black, temperature=temperature, max_plies=300)
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == (white is agent):
            wins += 1
        else:
            losses += 1
    return {"wins": wins, "draws": draws, "losses": losses}


def train(
    sl_ckpt: Path,
    ckpt_dir: Path,
    iterations: int = 200,
    games_per_iter: int = 16,
    epochs_per_iter: int = 2,
    batch_size: int = 256,
    lr: float = 1e-4,
    temperature: float = 0.7,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    eval_every: int = 10,
    eval_games: int = 10,
    device: str = "cpu",
    algo: str = "az",           # "az" = AlphaZero-style distillation, "ppo" = PPO
    mcts_sims: int = 64,        # sims per move during self-play (az only)
    az_temperature_drop_ply: int = 30,
    max_plies: int = 300,
    ppo_clip: float = 0.2,      # PPO clip range ε
    ppo_value_clip: float | None = 0.2,  # value clipping range; None disables
    normalize_advantage: bool = True,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    from .model import load_model

    model = load_model(
        sl_ckpt,
        device=dev,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Frozen baseline (= SL starting point) for regression checks.
    baseline = copy.deepcopy(model).to(dev)
    for p in baseline.parameters():
        p.requires_grad_(False)
    baseline.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for it in range(1, iterations + 1):
        t0 = time.time()

        # --- Phase 1: self-play ----------------------------------------------
        if algo == "az":
            trajs = [
                play_self_game_mcts(
                    model,
                    device=dev,
                    num_simulations=mcts_sims,
                    temperature=temperature,
                    temperature_drop_ply=az_temperature_drop_ply,
                    max_plies=max_plies,
                )
                for _ in range(games_per_iter)
            ]
        elif algo == "ppo":
            trajs = [
                play_self_game(model, device=dev, temperature=temperature, max_plies=max_plies)
                for _ in range(games_per_iter)
            ]
        else:
            raise ValueError(f"Unknown algo: {algo!r}")
        flat = _flatten_trajectories(trajs)
        batch = flat["batch"].to(dev)
        actions = flat["actions"].to(dev)
        returns = flat["returns"].to(dev)
        masks = flat["masks"].to(dev)
        pi_targets = flat["pi_targets"].to(dev) if flat["pi_targets"] is not None else None
        old_log_probs = flat["old_log_probs"].to(dev) if flat["old_log_probs"] is not None else None
        old_values = flat["old_values"].to(dev) if flat["old_values"] is not None else None
        n = actions.size(0)
        results = [t.result for t in trajs]
        decisive = sum(1 for r in results if r != "1/2-1/2")

        # Fixed advantages for PPO (recomputing per-epoch with current V would
        # introduce bias and defeat the point of the stop-grad baseline).
        if algo == "ppo":
            advantages = returns - old_values
            if normalize_advantage and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

        # --- Phase 2: policy gradient updates --------------------------------
        model.train()
        pg_acc = v_acc = ent_acc = kl_acc = clip_acc = 0.0
        pg_steps = 0
        for _ in range(epochs_per_iter):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                sub_batch = Batch.from_data_list([batch.get_example(int(i)) for i in idx])
                sub_batch = sub_batch.to(dev)
                sub_actions = actions[idx]
                sub_returns = returns[idx]
                sub_masks = masks[idx]

                policy, _, value = model(sub_batch)
                # Apply the same temperature used at collection time so the
                # new log-probs live in the same distribution family as the
                # stored old log-probs (keeps the PPO ratio unbiased).
                log_probs = masked_log_softmax(policy / max(temperature, 1e-6), sub_masks)

                if pi_targets is not None:
                    # AlphaZero-style distillation: cross-entropy to MCTS visits.
                    sub_pi = pi_targets[idx]
                    pg_loss = -(sub_pi * log_probs).sum(dim=-1).mean()
                    v_loss = F.mse_loss(value, sub_returns)
                    clip_frac = 0.0
                    approx_kl = 0.0
                else:
                    # PPO with clipped ratio.
                    lp_taken = log_probs.gather(1, sub_actions.unsqueeze(1)).squeeze(1)
                    sub_old_lp = old_log_probs[idx]
                    sub_adv = advantages[idx]
                    sub_old_v = old_values[idx]

                    ratio = torch.exp(lp_taken - sub_old_lp)
                    unclipped = ratio * sub_adv
                    clipped = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * sub_adv
                    pg_loss = -torch.min(unclipped, clipped).mean()

                    if ppo_value_clip is not None:
                        v_clipped = sub_old_v + torch.clamp(
                            value - sub_old_v, -ppo_value_clip, ppo_value_clip
                        )
                        v_loss_unclipped = (value - sub_returns).pow(2)
                        v_loss_clipped = (v_clipped - sub_returns).pow(2)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = F.mse_loss(value, sub_returns)

                    with torch.no_grad():
                        approx_kl = (sub_old_lp - lp_taken).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > ppo_clip).float().mean().item()

                # Entropy on legal moves only.
                probs = log_probs.exp()
                ent = -(probs * log_probs.clamp_min(torch.finfo(log_probs.dtype).min)).sum(dim=-1).mean()

                loss = pg_loss + value_coef * v_loss - entropy_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                pg_acc += pg_loss.item(); v_acc += v_loss.item(); ent_acc += ent.item()
                kl_acc += float(approx_kl); clip_acc += float(clip_frac); pg_steps += 1

        pg_steps = max(pg_steps, 1)
        dt = time.time() - t0
        if algo == "ppo":
            print(
                f"iter {it:>4d} | games {games_per_iter} | plies {n} | "
                f"decisive {decisive}/{games_per_iter} | "
                f"pg {pg_acc/pg_steps:.3f} v {v_acc/pg_steps:.3f} ent {ent_acc/pg_steps:.3f} "
                f"kl {kl_acc/pg_steps:.3f} clip {clip_acc/pg_steps:.2f} | {dt:.1f}s"
            )
        else:
            print(
                f"iter {it:>4d} | games {games_per_iter} | plies {n} | "
                f"decisive {decisive}/{games_per_iter} | "
                f"pg {pg_acc/pg_steps:.3f} v {v_acc/pg_steps:.3f} ent {ent_acc/pg_steps:.3f} | "
                f"{dt:.1f}s"
            )

        if it % eval_every == 0 or it == iterations:
            stats = evaluate(model, baseline, device=device, num_games=eval_games)
            print(f"  eval vs baseline: {stats}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "iter": it,
                    "config": model.config,
                },
                ckpt_dir / f"rl_iter{it}.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "iter": iterations,
            "config": model.config,
        },
        ckpt_dir / "rl_final.pt",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sl-ckpt", type=Path, required=True, help="Supervised checkpoint to fine-tune.")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/rl"))
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=16)
    p.add_argument("--epochs-per-iter", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--algo", choices=["az", "ppo"], default="az",
                   help="'az' uses MCTS + visit-distribution distillation (default); "
                        "'ppo' is PPO with a clipped ratio and value-head baseline.")
    p.add_argument("--mcts-sims", type=int, default=64, help="MCTS sims per move in self-play (az only).")
    p.add_argument("--az-temp-drop-ply", type=int, default=30)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--ppo-value-clip", type=float, default=0.2,
                   help="Value-function clip range; pass a negative value to disable.")
    p.add_argument("--no-normalize-advantage", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        sl_ckpt=args.sl_ckpt,
        ckpt_dir=args.ckpt,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        device=args.device,
        algo=args.algo,
        mcts_sims=args.mcts_sims,
        az_temperature_drop_ply=args.az_temp_drop_ply,
        max_plies=args.max_plies,
        ppo_clip=args.ppo_clip,
        ppo_value_clip=None if args.ppo_value_clip < 0 else args.ppo_value_clip,
        normalize_advantage=not args.no_normalize_advantage,
    )

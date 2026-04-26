"""Supervised pre-training on PGN data.

Usage:
    python -m chess_gnn.train_sl --pgn data/lichess_elite.pgn.zst \
        --steps 100000 --batch-size 256 --ckpt checkpoints/sl

Loss:
    main   = cross-entropy over legal moves (masked log-softmax on 4096 logits)
    aux    = cross-entropy on the 3-way under-promotion head, only on promoting moves
    value  = MSE on the final game result from the side-to-move's perspective
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Batch

from .dataset import EngineValueDataset, PGNMoveDataset
from .model import ChessGNN, masked_log_softmax


def _collate(data_list):
    return Batch.from_data_list(data_list)


def topk_accuracy(log_probs: torch.Tensor, target: torch.Tensor, k: int) -> float:
    topk = log_probs.topk(k, dim=-1).indices
    return (topk == target.unsqueeze(-1)).any(dim=-1).float().mean().item()


def train(
    pgn_paths: list[Path],
    ckpt_dir: Path,
    steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    min_elo: int | None = 2200,
    num_workers: int = 2,
    log_every: int = 50,
    ckpt_every: int = 2000,
    device: str = "cpu",
    resume: Path | None = None,
    value_coef: float = 0.25,
    engine_value_path: Path | None = None,
    engine_value_blend: float = 1.0,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    if engine_value_path is None:
        dataset = PGNMoveDataset(pgn_paths, shuffle_buffer=8192, min_elo=min_elo)
        shuffle = False
    else:
        dataset = EngineValueDataset(engine_value_path, value_blend=engine_value_blend)
        shuffle = True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )

    model = ChessGNN(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_step = 0
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        for group in opt.param_groups:
            group["lr"] = lr
        start_step = state.get("step", 0)
        print(f"resumed from {resume} @ step {start_step}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps - start_step, 1))

    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}

    step = start_step
    while step < steps:
        saw_batch = False
        for batch in loader:
            saw_batch = True
            if step >= steps:
                break
            batch = batch.to(dev)
            policy, underpromo, value = model(batch)

            mask = batch.legal_mask.view(-1, policy.size(-1))
            log_probs = masked_log_softmax(policy, mask)
            policy_loss = F.nll_loss(log_probs, batch.y)
            value_target = batch.value_target.view_as(value).to(dtype=value.dtype)
            value_loss = F.mse_loss(value, value_target)

            # Under-promo auxiliary loss over the batch's promoting moves only.
            up_target = batch.underpromo_target
            up_mask = up_target >= 0
            if up_mask.any():
                idx = batch.y[up_mask]
                batch_idx = torch.arange(policy.size(0), device=dev)[up_mask]
                up_logits = underpromo[batch_idx, idx, :]
                up_loss = F.cross_entropy(up_logits, up_target[up_mask])
            else:
                up_loss = torch.zeros((), device=dev)

            loss = policy_loss + 0.1 * up_loss + value_coef * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                running["loss"] += loss.item() * batch.y.size(0)
                running["value_loss"] += value_loss.item() * batch.y.size(0)
                running["top1"] += topk_accuracy(log_probs, batch.y, 1) * batch.y.size(0)
                running["top5"] += topk_accuracy(log_probs, batch.y, 5) * batch.y.size(0)
                running["n"] += batch.y.size(0)

            step += 1
            if step % log_every == 0:
                n = max(running["n"], 1)
                dt = time.time() - t0
                print(
                    f"step {step:>7d} | loss {running['loss']/n:.4f} "
                    f"| v {running['value_loss']/n:.4f} "
                    f"| top1 {running['top1']/n:.3f} | top5 {running['top5']/n:.3f} "
                    f"| {n / dt:.1f} pos/s"
                )
                running = {"loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
                t0 = time.time()

            if step % ckpt_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "config": model.config,
                    },
                    ckpt_dir / f"sl_step{step}.pt",
                )
        if isinstance(dataset, IterableDataset) or not saw_batch:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "config": model.config,
        },
        ckpt_dir / "sl_final.pt",
    )
    print(f"done @ step {step}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", type=Path, nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sl"))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--min-elo", type=int, default=2200)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--value-coef", type=float, default=0.25)
    p.add_argument("--engine-value-data", type=Path, default=None)
    p.add_argument(
        "--engine-value-blend",
        type=float,
        default=1.0,
        help="Blend for engine target vs game target when --engine-value-data is used.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        pgn_paths=args.pgn,
        ckpt_dir=args.ckpt,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        min_elo=args.min_elo,
        num_workers=args.num_workers,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        device=args.device,
        resume=args.resume,
        value_coef=args.value_coef,
        engine_value_path=args.engine_value_data,
        engine_value_blend=args.engine_value_blend,
    )

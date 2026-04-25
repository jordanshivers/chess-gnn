"""GNN chess model.

Trunk: a stack of `TransformerConv` layers over the fully-connected 64-node
graph, with residual connections and LayerNorm. Edge features (geometric
priors) are fed in via PyG's `edge_dim` plumbing.

Heads:
  * policy: one logit per directed (from, to) edge, flattened to 4096.
  * under-promotion: per-edge 3-way logits (N/B/R), consulted only when a
    promoting-pawn move is selected.
  * value: scalar in [-1, 1], useful as an RL baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch

from .encoding import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, NUM_SQUARES
from .moves import NUM_MOVES


class ChessGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        self.node_in = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=EDGE_FEATURE_DIM,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.policy_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + EDGE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.underpromo_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + EDGE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self, data: Data | Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B, 4096], underpromo_logits [B, 4096, 3], value [B])."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.node_in(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.gelu(layer(h, edge_index, edge_attr)))

        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(
            h.size(0), dtype=torch.long, device=h.device
        )
        # Per-graph dense node tensors: [B, 64, hidden]
        h_dense, _ = to_dense_batch(h, batch, max_num_nodes=NUM_SQUARES)
        B, N, H = h_dense.shape
        assert N == NUM_SQUARES

        # Build per-edge features over the fully-connected graph, reused across batch.
        from_idx = torch.arange(N, device=h.device).repeat_interleave(N)   # [4096]
        to_idx = torch.arange(N, device=h.device).repeat(N)                # [4096]
        # Edge attr for all 4096 pairs (including self-loops, which we mask out).
        # We reuse the precomputed edge_attr that skipped self-loops; rebuild dense.
        pair_edge_attr = self._build_full_pair_edge_attr(h.device)          # [4096, E_dim]
        pair_edge_attr = pair_edge_attr.unsqueeze(0).expand(B, -1, -1)      # [B, 4096, E_dim]

        h_from = h_dense[:, from_idx, :]                                    # [B, 4096, H]
        h_to = h_dense[:, to_idx, :]                                        # [B, 4096, H]
        pair_in = torch.cat([h_from, h_to, pair_edge_attr], dim=-1)         # [B, 4096, 2H+E]

        policy_logits = self.policy_head(pair_in).squeeze(-1)               # [B, 4096]
        underpromo_logits = self.underpromo_head(pair_in)                   # [B, 4096, 3]

        # Self-loop indices (from == to) -> mask to -inf so they never get picked.
        self_loop = (from_idx == to_idx)
        policy_logits = policy_logits.masked_fill(self_loop, float("-inf"))

        graph_embed = h_dense.mean(dim=1)                                   # [B, H]
        value = self.value_head(graph_embed).squeeze(-1)                    # [B]
        return policy_logits, underpromo_logits, value

    # ----- helpers ------------------------------------------------------------

    _PAIR_EDGE_ATTR: torch.Tensor | None = None

    @classmethod
    def _build_full_pair_edge_attr(cls, device: torch.device) -> torch.Tensor:
        """Per-pair (64*64=4096) edge attributes, matching the policy index order.

        Diagonal (from == to) slots hold zeros; they are masked out in forward.
        """
        cached = cls._PAIR_EDGE_ATTR
        if cached is not None and cached.device == device:
            return cached
        import chess  # local import to avoid top-level coupling

        N = NUM_SQUARES
        attr = torch.zeros(N * N, EDGE_FEATURE_DIM, device=device)
        for i in range(N):
            fi, ri = chess.square_file(i), chess.square_rank(i)
            for j in range(N):
                if i == j:
                    continue
                fj, rj = chess.square_file(j), chess.square_rank(j)
                df, dr = fj - fi, rj - ri
                adf, adr = abs(df), abs(dr)
                attr[i * N + j] = torch.tensor(
                    [
                        df / 7.0,
                        dr / 7.0,
                        float(dr == 0),
                        float(df == 0),
                        float(adf == adr),
                        float((adf, adr) in ((1, 2), (2, 1))),
                        max(adf, adr) / 7.0,
                    ],
                    device=device,
                )
        cls._PAIR_EDGE_ATTR = attr
        return attr


def _infer_arch_from_state(state_dict: dict) -> dict:
    """Recover hidden_dim and num_layers from state-dict shapes.

    `num_heads` is not recoverable from weights alone (TransformerConv uses
    `concat=True`, so per-layer Linear shapes are independent of head count).
    """
    hidden_dim = int(state_dict["node_in.weight"].shape[0])
    layer_indices = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
    }
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    return {"hidden_dim": hidden_dim, "num_layers": num_layers}


def load_model(
    ckpt_path,
    device: str | torch.device = "cpu",
    **fallback_config,
) -> "ChessGNN":
    """Instantiate `ChessGNN` and load weights, recovering arch config when possible.

    Resolution order for each config field:
      1. embedded `config` dict in the checkpoint (preferred; new checkpoints have this)
      2. shape-inferred `hidden_dim` / `num_layers` from the state dict
      3. `fallback_config` kwargs (used for `num_heads`, which is not inferable)
      4. `ChessGNN` defaults

    `num_heads` defaults to 4 when absent; if that doesn't divide the detected
    `hidden_dim`, we step down to the largest divisor in {4, 2, 1}. Pass an
    explicit `num_heads=` to override.
    """
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"]
    embedded = state.get("config") or {}
    inferred = _infer_arch_from_state(sd)

    cfg: dict = {}
    cfg["hidden_dim"] = (
        embedded.get("hidden_dim")
        or fallback_config.get("hidden_dim")
        or inferred["hidden_dim"]
    )
    cfg["num_layers"] = (
        embedded.get("num_layers")
        or fallback_config.get("num_layers")
        or inferred["num_layers"]
    )
    nh = embedded.get("num_heads") or fallback_config.get("num_heads") or 4
    if cfg["hidden_dim"] % nh != 0:
        for candidate in (4, 2, 1):
            if cfg["hidden_dim"] % candidate == 0:
                print(f"[load_model] num_heads={nh} doesn't divide hidden_dim={cfg['hidden_dim']}; "
                      f"using num_heads={candidate}")
                nh = candidate
                break
    cfg["num_heads"] = nh
    if "dropout" in embedded:
        cfg["dropout"] = embedded["dropout"]

    if not embedded:
        print(f"[load_model] no embedded config in {ckpt_path}; using arch={cfg} "
              f"(hidden_dim/num_layers inferred from weights; num_heads assumed)")

    model = ChessGNN(**cfg)
    model.load_state_dict(sd)
    model.to(device)
    return model


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal moves only. `mask` is a [*, 4096] bool tensor."""
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    return F.log_softmax(logits, dim=-1)

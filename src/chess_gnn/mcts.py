"""PUCT Monte-Carlo Tree Search over the policy/value network.

Standard AlphaZero-style PUCT with no rollouts: leaf value comes from the
value head, priors from the policy head. Single-threaded, designed for
clarity first; a few hundred simulations per move is the intended regime.

Usage (see `play.py`):
    mcts = MCTS(model, device, c_puct=1.5)
    move = mcts.search(board, num_simulations=200, temperature=0.1)

Set `num_simulations=0` on the agent to disable and fall back to the raw
policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import chess
import torch
from torch_geometric.data import Batch

from .encoding import board_to_data
from .model import ChessGNN
from .moves import decode_move, legal_mask


@dataclass
class _Node:
    """One position in the tree. Edges are keyed by move-index (0..4095)."""
    to_play: bool                      # chess.WHITE or chess.BLACK at this node
    prior: dict[int, float] = field(default_factory=dict)   # P(s, a)
    visits: dict[int, int] = field(default_factory=dict)    # N(s, a)
    value_sum: dict[int, float] = field(default_factory=dict)  # W(s, a), from to_play's POV
    children: dict[int, "_Node"] = field(default_factory=dict)
    total_visits: int = 0
    is_terminal: bool = False
    terminal_value: float = 0.0        # from to_play's POV


class MCTS:
    """PUCT search using a ChessGNN for priors + leaf values.

    `c_puct` is the exploration constant. `dirichlet_alpha` / `dirichlet_eps`
    add Dirichlet noise to the root priors when > 0 (useful during self-play
    for exploration; leave at 0 for evaluation / play).
    """

    def __init__(
        self,
        model: ChessGNN,
        device: str | torch.device = "cpu",
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.0,
        dirichlet_eps: float = 0.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    # --- public API -----------------------------------------------------------

    @torch.no_grad()
    def search(
        self,
        board: chess.Board,
        num_simulations: int,
        temperature: float = 0.0,
    ) -> chess.Move:
        """Run `num_simulations` and return a move chosen from the visit counts.

        `temperature` applies to the N^(1/T) distribution over children:
          - T <= 1e-3: argmax visits (deterministic)
          - otherwise: sample from pi(a) ∝ N(s,a)^{1/T}
        """
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"No legal moves in {board.fen()}")

        root = _Node(to_play=board.turn)
        self._expand(root, board, add_dirichlet=(self.dirichlet_eps > 0.0))

        for _ in range(num_simulations):
            self._simulate(root, board.copy(stack=False))

        # Pick a move from visit counts.
        items = [(a, n) for a, n in root.visits.items() if n > 0]
        if not items:
            # Degenerate (no sims, or every legal child is terminal loss).
            # Fall back to the argmax prior.
            a = max(root.prior, key=root.prior.get)
            return _action_to_move(board, a)

        if temperature <= 1e-3:
            a = max(items, key=lambda x: x[1])[0]
        else:
            acts = torch.tensor([a for a, _ in items], dtype=torch.long)
            counts = torch.tensor([float(n) for _, n in items])
            weights = counts.pow(1.0 / temperature)
            weights = weights / weights.sum().clamp_min(1e-12)
            a = int(acts[torch.multinomial(weights, 1).item()].item())
        return _action_to_move(board, a)

    @torch.no_grad()
    def visit_distribution(
        self, board: chess.Board, num_simulations: int
    ) -> tuple[list[chess.Move], list[float]]:
        """Expose the root visit counts as a distribution — useful for SL
        distillation or inspection (tests, viz)."""
        root = _Node(to_play=board.turn)
        self._expand(root, board, add_dirichlet=False)
        for _ in range(num_simulations):
            self._simulate(root, board.copy(stack=False))
        items = [(a, n) for a, n in root.visits.items() if n > 0]
        total = sum(n for _, n in items) or 1
        moves = [_action_to_move(board, a) for a, _ in items]
        probs = [n / total for _, n in items]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        return [moves[i] for i in order], [probs[i] for i in order]

    # --- internals ------------------------------------------------------------

    def _simulate(self, root: _Node, board: chess.Board) -> None:
        """One PUCT descent: select -> expand leaf -> backup."""
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if node.is_terminal:
                self._backup(path, node.terminal_value, node.to_play)
                return
            a = self._select_action(node)
            path.append((node, a))
            if a not in node.children:
                # New edge: create the child, evaluate, backup.
                board.push(_action_to_move(board, a))
                child = _Node(to_play=board.turn)
                node.children[a] = child
                leaf_value = self._expand(child, board, add_dirichlet=False)
                self._backup(path, leaf_value, child.to_play)
                return
            board.push(_action_to_move(board, a))
            node = node.children[a]

    def _select_action(self, node: _Node) -> int:
        """PUCT: argmax Q + c_puct * P * sqrt(N_total) / (1 + N(a))."""
        sqrt_total = math.sqrt(max(node.total_visits, 1))
        best_a, best_score = -1, -float("inf")
        for a, p in node.prior.items():
            n = node.visits.get(a, 0)
            q = (node.value_sum.get(a, 0.0) / n) if n > 0 else 0.0
            u = self.c_puct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _backup(self, path: list[tuple[_Node, int]], leaf_value: float, leaf_to_play: bool) -> None:
        """Propagate `leaf_value` (from leaf_to_play's POV) up the path,
        flipping sign whenever the node's side-to-move differs from the leaf's."""
        for node, a in reversed(path):
            v = leaf_value if node.to_play == leaf_to_play else -leaf_value
            node.visits[a] = node.visits.get(a, 0) + 1
            node.value_sum[a] = node.value_sum.get(a, 0.0) + v
            node.total_visits += 1

    @torch.no_grad()
    def _expand(self, node: _Node, board: chess.Board, add_dirichlet: bool) -> float:
        """Evaluate `board` with the net, store priors + value on `node`.

        Returns the value estimate from `node.to_play`'s POV (used for backup).
        If the position is terminal, sets `node.is_terminal` and returns the
        game result from that perspective.
        """
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            # White-POV outcome.
            w = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0)
            v = w if node.to_play == chess.WHITE else -w
            node.is_terminal = True
            node.terminal_value = v
            return v

        data = board_to_data(board)
        batch = Batch.from_data_list([data]).to(self.device)
        policy, _, value = self.model(batch)
        mask = legal_mask(board).to(self.device).unsqueeze(0)
        neg_inf = torch.finfo(policy.dtype).min
        masked = policy.masked_fill(~mask, neg_inf)
        probs = torch.softmax(masked, dim=-1).squeeze(0).cpu()
        legal_idx = mask.squeeze(0).nonzero().flatten().tolist()

        # Prior: P(a) for each legal action-index. We key by (from,to) index
        # so the tree is promotion-agnostic; decode_move defaults to queen.
        priors = {a: float(probs[a].item()) for a in legal_idx}
        total = sum(priors.values()) or 1.0
        priors = {a: p / total for a, p in priors.items()}

        if add_dirichlet and self.dirichlet_alpha > 0 and priors:
            alpha = [self.dirichlet_alpha] * len(priors)
            noise = torch.distributions.Dirichlet(torch.tensor(alpha)).sample().tolist()
            eps = self.dirichlet_eps
            keys = list(priors.keys())
            for k, n in zip(keys, noise):
                priors[k] = (1 - eps) * priors[k] + eps * n

        node.prior = priors
        # The value head is a tanh in [-1, 1] — we assume it's from the side-
        # to-move's POV (it's used that way in train_rl.py's baseline).
        return float(value.item())


def _action_to_move(board: chess.Board, action: int) -> chess.Move:
    """Decode a policy index to a legal `chess.Move`, defaulting promotions to queen.

    Falls back to the first legal move with matching (from,to) squares if
    `decode_move` refuses the combination — robustness for odd edge cases
    near promotion squares.
    """
    try:
        return decode_move(board, action, underpromo_choice=None)
    except ValueError:
        f, t = divmod(action, 64)
        for m in board.legal_moves:
            if m.from_square == f and m.to_square == t:
                return m
        raise


__all__ = ["MCTS"]

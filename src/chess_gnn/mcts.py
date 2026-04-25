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


@dataclass
class _PendingEval:
    """A newly-created non-terminal leaf waiting for batched network eval."""

    node: _Node
    board: chess.Board
    path: list[tuple[_Node, int]]
    virtual_loss: float = 1.0


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
        self.virtual_loss = 1.0
        self.root: _Node | None = None
        self.root_key: str | None = None

    # --- public API -----------------------------------------------------------

    @torch.no_grad()
    def search(
        self,
        board: chess.Board,
        num_simulations: int,
        temperature: float = 0.0,
        batch_size: int = 1,
    ) -> chess.Move:
        """Run `num_simulations` and return a move chosen from the visit counts.

        `temperature` applies to the N^(1/T) distribution over children:
          - T <= 1e-3: argmax visits (deterministic)
          - otherwise: sample from pi(a) ∝ N(s,a)^{1/T}
        """
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"No legal moves in {board.fen()}")

        root = self._new_root(board, add_dirichlet=(self.dirichlet_eps > 0.0))
        self._run_simulations(root, board, num_simulations, batch_size=batch_size)

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
        self,
        board: chess.Board,
        num_simulations: int,
        add_dirichlet: bool | None = None,
        batch_size: int = 1,
    ) -> tuple[list[chess.Move], list[float]]:
        """Expose the root visit counts as a distribution — useful for SL
        distillation or inspection (tests, viz).

        `add_dirichlet` controls root exploration noise. By default it follows
        this searcher's configuration; callers doing deterministic evaluation
        should pass False explicitly.
        """
        use_noise = (self.dirichlet_eps > 0.0) if add_dirichlet is None else add_dirichlet
        root = self._new_root(board, add_dirichlet=use_noise)
        self._run_simulations(root, board, num_simulations, batch_size=batch_size)
        return self._visit_distribution_from_root(root, board)

    @torch.no_grad()
    def prepare_root(self, board: chess.Board, add_dirichlet: bool = False) -> None:
        """Ensure the reusable root matches `board`, expanding it if needed."""
        key = _board_key(board)
        if self.root is not None and self.root_key == key:
            return
        self.root = self._new_root(board, add_dirichlet=add_dirichlet)
        self.root_key = key

    def advance_root(self, move: chess.Move, board_after: chess.Board) -> None:
        """Reuse the chosen child as the next root after `move` has been pushed."""
        action = move.from_square * 64 + move.to_square
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root_key = _board_key(board_after)
            return
        # Fallback for externally supplied moves that were not searched.
        self.root = self._new_root(board_after, add_dirichlet=False)
        self.root_key = _board_key(board_after)

    def add_root_dirichlet_noise(self) -> None:
        """Mix configured Dirichlet noise into the reusable root priors."""
        if self.root is not None:
            self._add_dirichlet_noise(self.root.prior)

    def run_simulation_round(self, board: chess.Board) -> _PendingEval | None:
        """Run selection for one simulation, returning a leaf to batch-evaluate."""
        self.prepare_root(board, add_dirichlet=False)
        assert self.root is not None
        return self._select_pending(self.root, board.copy(stack=False))

    def collect_simulation_batch(
        self,
        board: chess.Board,
        num_simulations: int,
        batch_size: int,
    ) -> list[_PendingEval]:
        """Collect up to `batch_size` pending leaves from this tree.

        Virtual loss is applied as leaves are selected, so repeated descents can
        fan out within one tree before a batched network evaluation.
        """
        pending: list[_PendingEval] = []
        for _ in range(min(num_simulations, batch_size)):
            leaf = self.run_simulation_round(board)
            if leaf is not None:
                pending.append(leaf)
        return pending

    def root_visit_distribution(self, board: chess.Board) -> tuple[list[chess.Move], list[float]]:
        """Return the reusable root's visit distribution for `board`."""
        self.prepare_root(board, add_dirichlet=False)
        assert self.root is not None
        return self._visit_distribution_from_root(self.root, board)

    @torch.no_grad()
    def evaluate_pending_batch(self, pending: list[_PendingEval]) -> None:
        """Evaluate pending leaves in one model call and back up their values."""
        if not pending:
            return

        data_list = [board_to_data(p.board) for p in pending]
        batch = Batch.from_data_list(data_list).to(self.device)
        policy, _, value = self.model(batch)
        masks = torch.stack([legal_mask(p.board) for p in pending], dim=0).to(self.device)
        neg_inf = torch.finfo(policy.dtype).min
        masked = policy.masked_fill(~masks, neg_inf)
        probs = torch.softmax(masked, dim=-1).cpu()
        masks_cpu = masks.cpu()
        values = value.detach().cpu().tolist()

        for i, p in enumerate(pending):
            legal_idx = masks_cpu[i].nonzero().flatten().tolist()
            priors = {a: float(probs[i, a].item()) for a in legal_idx}
            total = sum(priors.values()) or 1.0
            p.node.prior = {a: prob / total for a, prob in priors.items()}
            self._revert_virtual_loss(p.path, p.virtual_loss)
            self._backup(p.path, float(values[i]), p.node.to_play)

    def select_root_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        """Choose a move from the reusable root's visit counts."""
        self.prepare_root(board, add_dirichlet=False)
        assert self.root is not None
        items = [(a, n) for a, n in self.root.visits.items() if n > 0]
        if not items:
            a = max(self.root.prior, key=self.root.prior.get)
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

    # --- internals ------------------------------------------------------------

    def _new_root(self, board: chess.Board, add_dirichlet: bool) -> _Node:
        root = _Node(to_play=board.turn)
        self._expand(root, board, add_dirichlet=add_dirichlet)
        return root

    def _run_simulations(
        self,
        root: _Node,
        board: chess.Board,
        num_simulations: int,
        batch_size: int = 1,
    ) -> None:
        sims_done = 0
        batch_size = max(1, int(batch_size))
        while sims_done < num_simulations:
            chunk = min(batch_size, num_simulations - sims_done)
            pending = []
            for _ in range(chunk):
                leaf = self._select_pending(root, board.copy(stack=False))
                if leaf is not None:
                    pending.append(leaf)
            sims_done += chunk
            if pending:
                self.evaluate_pending_batch(pending)

    def _visit_distribution_from_root(
        self, root: _Node, board: chess.Board
    ) -> tuple[list[chess.Move], list[float]]:
        items = [(a, n) for a, n in root.visits.items() if n > 0]
        total = sum(n for _, n in items) or 1
        moves = [_action_to_move(board, a) for a, _ in items]
        probs = [n / total for _, n in items]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        return [moves[i] for i in order], [probs[i] for i in order]

    def _select_pending(self, root: _Node, board: chess.Board) -> _PendingEval | None:
        """One PUCT descent: select -> create a leaf for later eval -> backup."""
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if node.is_terminal:
                self._backup(path, node.terminal_value, node.to_play)
                return None
            a = self._select_action(node)
            if a is None:
                return None
            path.append((node, a))
            if a not in node.children:
                board.push(_action_to_move(board, a))
                child = _Node(to_play=board.turn)
                node.children[a] = child
                if board.is_game_over(claim_draw=True):
                    value = _terminal_value(board, child.to_play)
                    child.is_terminal = True
                    child.terminal_value = value
                    self._backup(path, value, child.to_play)
                    return None
                self._apply_virtual_loss(path, self.virtual_loss)
                return _PendingEval(
                    node=child,
                    board=board.copy(stack=False),
                    path=path,
                    virtual_loss=self.virtual_loss,
                )
            board.push(_action_to_move(board, a))
            node = node.children[a]

    def _select_action(self, node: _Node) -> int | None:
        """PUCT: argmax Q + c_puct * P * sqrt(N_total) / (1 + N(a))."""
        sqrt_total = math.sqrt(max(node.total_visits, 1))
        best_a, best_score = None, -float("inf")
        for a, p in node.prior.items():
            child = node.children.get(a)
            if child is not None and not child.is_terminal and not child.prior:
                # Child has been selected for batched evaluation but not
                # expanded yet; virtual loss already reserves this path.
                continue
            n = node.visits.get(a, 0)
            q = (node.value_sum.get(a, 0.0) / n) if n > 0 else 0.0
            u = self.c_puct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _apply_virtual_loss(self, path: list[tuple[_Node, int]], virtual_loss: float) -> None:
        for node, a in reversed(path):
            node.visits[a] = node.visits.get(a, 0) + 1
            node.value_sum[a] = node.value_sum.get(a, 0.0) - virtual_loss
            node.total_visits += 1

    def _revert_virtual_loss(self, path: list[tuple[_Node, int]], virtual_loss: float) -> None:
        for node, a in reversed(path):
            node.visits[a] = node.visits.get(a, 0) - 1
            if node.visits[a] <= 0:
                del node.visits[a]
            node.value_sum[a] = node.value_sum.get(a, 0.0) + virtual_loss
            if abs(node.value_sum[a]) < 1e-12:
                del node.value_sum[a]
            node.total_visits -= 1

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

        if add_dirichlet:
            self._add_dirichlet_noise(priors)

        node.prior = priors
        # The value head is a tanh in [-1, 1] — we assume it's from the side-
        # to-move's POV (it's used that way in train_rl.py's baseline).
        return float(value.item())

    def _add_dirichlet_noise(self, priors: dict[int, float]) -> None:
        if self.dirichlet_alpha <= 0 or self.dirichlet_eps <= 0 or not priors:
            return
        alpha = [self.dirichlet_alpha] * len(priors)
        noise = torch.distributions.Dirichlet(torch.tensor(alpha)).sample().tolist()
        eps = self.dirichlet_eps
        keys = list(priors.keys())
        for k, n in zip(keys, noise):
            priors[k] = (1 - eps) * priors[k] + eps * n


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


def _board_key(board: chess.Board) -> str:
    return board.fen()


def _terminal_value(board: chess.Board, to_play: bool) -> float:
    res = board.result(claim_draw=True)
    w = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0)
    return w if to_play == chess.WHITE else -w


__all__ = ["MCTS"]

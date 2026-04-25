"""Agent that turns a trained model into a chess player."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from torch_geometric.data import Batch

from .encoding import board_to_data
from .mcts import MCTS
from .model import ChessGNN
from .moves import decode_move, legal_mask


@dataclass
class MoveRanking:
    """A ranked list of legal moves with their probabilities, for inspection/viz."""

    moves: list[chess.Move]
    probabilities: list[float]


class GNNAgent:
    """Wraps a trained model as a chess-playing agent.

    Two move-selection modes:
      * **Raw policy** (default, `num_simulations=0`): sample directly from the
        temperature-scaled policy head. Fast; the baseline for all eval so far.
      * **MCTS** (`num_simulations > 0`): PUCT search using the policy head as
        priors and the value head at leaves. Roughly one forward pass per
        batch of simulations; `mcts_batch_size` trades a bit of search
        synchrony for much better accelerator utilization.

    `num_simulations` can also be passed per-call to `rank_moves` / `select_move`
    to override the default.
    """

    def __init__(
        self,
        model: ChessGNN,
        device: str | torch.device = "cpu",
        default_temperature: float = 1.0,
        num_simulations: int = 0,
        mcts_batch_size: int = 1,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.0,
        dirichlet_eps: float = 0.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.default_temperature = default_temperature
        self.num_simulations = int(num_simulations)
        self.mcts_batch_size = max(1, int(mcts_batch_size))
        self._mcts = MCTS(
            self.model,
            device=self.device,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )

    def _resolve_sims(self, num_simulations: int | None) -> int:
        return self.num_simulations if num_simulations is None else int(num_simulations)

    @torch.no_grad()
    def rank_moves(
        self,
        board: chess.Board,
        temperature: float | None = None,
        num_simulations: int | None = None,
    ) -> MoveRanking:
        """Rank legal moves in `board`.

        With `num_simulations > 0`, the ranking is over MCTS visit counts
        (the priors used by the search itself come from the raw policy).
        Otherwise the ranking is the temperature-scaled raw policy.
        """
        sims = self._resolve_sims(num_simulations)
        if sims > 0 and not board.is_game_over(claim_draw=True):
            moves, probs = self._mcts.visit_distribution(
                board,
                sims,
                add_dirichlet=False,
                batch_size=self.mcts_batch_size,
            )
            return MoveRanking(moves=moves, probabilities=probs)

        temperature = temperature if temperature is not None else self.default_temperature
        data = board_to_data(board)
        batch = Batch.from_data_list([data]).to(self.device)
        policy, underpromo, _ = self.model(batch)
        mask = legal_mask(board).to(self.device).unsqueeze(0)

        scaled = policy / max(temperature, 1e-6)
        neg_inf = torch.finfo(scaled.dtype).min
        scaled = scaled.masked_fill(~mask, neg_inf)
        probs = torch.softmax(scaled, dim=-1).squeeze(0)

        legal_idx = mask.squeeze(0).nonzero().flatten().tolist()
        move_probs: list[tuple[chess.Move, float]] = []
        for idx in legal_idx:
            # If the index is a promoting-pawn move, split it across under/queen options.
            try:
                queen_move = decode_move(board, idx, underpromo_choice=None)
            except ValueError:
                queen_move = None
            if queen_move is not None and queen_move.promotion == chess.QUEEN:
                up_logits = underpromo[0, idx]
                # Weight: queen gets the base probability; under-promos split an auxiliary mass.
                # Simple scheme: treat queen as default, under-promotions as minor perturbations.
                up_probs = torch.softmax(up_logits, dim=-1).tolist()
                # Allocate 20% of the cell to under-promotions, 80% to queen, as a neutral prior.
                cell = probs[idx].item()
                move_probs.append((queen_move, cell * 0.8))
                for up_choice, piece in enumerate((chess.KNIGHT, chess.BISHOP, chess.ROOK)):
                    try:
                        m = decode_move(board, idx, underpromo_choice=up_choice)
                    except ValueError:
                        continue
                    move_probs.append((m, cell * 0.2 * up_probs[up_choice]))
            elif queen_move is not None:
                move_probs.append((queen_move, probs[idx].item()))

        move_probs.sort(key=lambda x: x[1], reverse=True)
        return MoveRanking(
            moves=[m for m, _ in move_probs],
            probabilities=[p for _, p in move_probs],
        )

    def select_move(
        self,
        board: chess.Board,
        temperature: float | None = None,
        num_simulations: int | None = None,
    ) -> chess.Move:
        t = temperature if temperature is not None else self.default_temperature
        sims = self._resolve_sims(num_simulations)
        if sims > 0 and not board.is_game_over(claim_draw=True):
            # Let MCTS do its own visit-count temperature sampling; the raw
            # policy's temperature doesn't apply here.
            return self._mcts.search(
                board,
                num_simulations=sims,
                temperature=t,
                batch_size=self.mcts_batch_size,
            )

        ranking = self.rank_moves(board, temperature=temperature, num_simulations=0)
        if not ranking.moves:
            raise ValueError(f"No legal moves in position {board.fen()}")
        if t <= 1e-4:
            return ranking.moves[0]
        probs = torch.tensor(ranking.probabilities, dtype=torch.float32)
        probs = probs / probs.sum().clamp_min(1e-12)
        choice = int(torch.multinomial(probs, num_samples=1).item())
        return ranking.moves[choice]


def play_game(
    white: GNNAgent,
    black: GNNAgent,
    temperature: float = 0.3,
    max_plies: int = 400,
) -> tuple[chess.Board, str]:
    """Play a single game between two agents. Returns (final board, result string)."""
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over():
            break
        agent = white if board.turn == chess.WHITE else black
        board.push(agent.select_move(board, temperature=temperature))
    return board, board.result(claim_draw=True)

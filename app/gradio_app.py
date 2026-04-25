"""Gradio play/inspection app for a trained chess-gnn checkpoint.

Run locally:
    python -m app.gradio_app --ckpt checkpoints/sl/sl_final.pt

Features:
- Paste or edit a FEN; the board updates with predicted-move arrows.
- Temperature slider controls how peaked the softmax is when sampling.
- "Play top move" / "Sample a move" step the game forward.
- "Reset" restores the starting position.
- "Undo" pops the last half-move.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.svg
import gradio as gr
import torch

from chess_gnn.model import ChessGNN, load_model
from chess_gnn.play import GNNAgent
from chess_gnn.viz import render_prediction_svg


def _wrap_svg(svg: str) -> str:
    # Gradio's HTML renderer needs a container; also center the board.
    return f"<div style='display:flex;justify-content:center'>{svg}</div>"


def build_app(agent: GNNAgent) -> gr.Blocks:
    with gr.Blocks(title="chess-gnn") as demo:
        gr.Markdown("# chess-gnn\nGNN chess engine — inspect predictions and play.")

        fen_state = gr.State(chess.STARTING_FEN)
        history_state = gr.State([])  # list[str] of FENs for undo

        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML()
            with gr.Column(scale=1):
                fen_box = gr.Textbox(value=chess.STARTING_FEN, label="FEN", lines=2)
                temperature = gr.Slider(0.01, 2.0, value=0.3, step=0.01, label="Temperature")
                topk = gr.Slider(1, 15, value=6, step=1, label="Top-k arrows")
                move_table = gr.Dataframe(
                    headers=["move (uci)", "p"],
                    datatype=["str", "number"],
                    label="Top predicted moves",
                    interactive=False,
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    top_btn = gr.Button("Play top")
                    sample_btn = gr.Button("Sample")
                with gr.Row():
                    undo_btn = gr.Button("Undo")
                    reset_btn = gr.Button("Reset")
                status = gr.Markdown()

        def _render(fen: str, temperature: float, topk: int):
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return "", [], f"**Invalid FEN:** {e}"
            if board.is_game_over(claim_draw=True):
                svg = chess.svg.board(board, size=420)
                return _wrap_svg(svg), [], f"**Game over:** {board.result(claim_draw=True)}"
            ranking = agent.rank_moves(board, temperature=temperature)
            rows = [[m.uci(), round(p, 4)] for m, p in list(zip(ranking.moves, ranking.probabilities))[:int(topk)]]
            svg = render_prediction_svg(board, ranking, topk=int(topk))
            turn = "white" if board.turn == chess.WHITE else "black"
            return _wrap_svg(svg), rows, f"{turn} to move"

        def _predict(fen, temp, k):
            html, rows, msg = _render(fen, temp, k)
            return html, rows, msg

        def _step(fen, temp, k, history, sample: bool):
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return fen, fen, "", [], history, f"**Invalid FEN:** {e}"
            if board.is_game_over(claim_draw=True):
                html, rows, msg = _render(fen, temp, k)
                return fen, fen, html, rows, history, msg
            move = agent.select_move(board, temperature=temp if sample else 0.0)
            history = history + [fen]
            board.push(move)
            new_fen = board.fen()
            html, rows, msg = _render(new_fen, temp, k)
            msg = f"played **{move.uci()}** — " + msg
            return new_fen, new_fen, html, rows, history, msg

        def _undo(fen, history, temp, k):
            if not history:
                html, rows, msg = _render(fen, temp, k)
                return fen, fen, html, rows, history, msg
            prev = history[-1]
            html, rows, msg = _render(prev, temp, k)
            return prev, prev, html, rows, history[:-1], msg

        def _reset(temp, k):
            fen = chess.STARTING_FEN
            html, rows, msg = _render(fen, temp, k)
            return fen, fen, html, rows, [], msg

        predict_btn.click(_predict, [fen_box, temperature, topk], [board_html, move_table, status])
        top_btn.click(
            lambda fen, t, k, h: _step(fen, t, k, h, sample=False),
            [fen_box, temperature, topk, history_state],
            [fen_box, fen_state, board_html, move_table, history_state, status],
        )
        sample_btn.click(
            lambda fen, t, k, h: _step(fen, t, k, h, sample=True),
            [fen_box, temperature, topk, history_state],
            [fen_box, fen_state, board_html, move_table, history_state, status],
        )
        undo_btn.click(
            _undo,
            [fen_box, history_state, temperature, topk],
            [fen_box, fen_state, board_html, move_table, history_state, status],
        )
        reset_btn.click(
            _reset,
            [temperature, topk],
            [fen_box, fen_state, board_html, move_table, history_state, status],
        )

        demo.load(_predict, [fen_box, temperature, topk], [board_html, move_table, status])

    return demo


def _load_agent(
    ckpt: Path,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    device: str,
) -> GNNAgent:
    if ckpt.exists():
        model = load_model(
            ckpt,
            device=device,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        print(f"loaded {ckpt} (config={model.config})")
    else:
        print(f"[warn] no checkpoint at {ckpt} — using randomly initialized model")
        model = ChessGNN(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)
    return GNNAgent(model, device=device, default_temperature=0.3)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sl/sl_final.pt"))
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agent = _load_agent(args.ckpt, args.hidden_dim, args.num_layers, args.num_heads, args.device)
    demo = build_app(agent)
    demo.launch(share=args.share)

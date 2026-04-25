"""Gradio app: play a full game against a trained chess-gnn checkpoint.

Run locally:
    python -m app.gradio_play --ckpt checkpoints/rl/rl_final.pt

The human picks a color and a temperature, selects a legal move from a
dropdown (or types a UCI/SAN move), and the agent replies automatically.
The board shows the agent's last move and, optionally, arrows for its
top-k predicted moves on the current position.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.svg
import gradio as gr

from chess_gnn.model import ChessGNN, load_model
from chess_gnn.play import GNNAgent
from chess_gnn.viz import render_prediction_svg


def _wrap_svg(svg: str) -> str:
    return f"<div style='display:flex;justify-content:center'>{svg}</div>"


def _legal_uci(board: chess.Board) -> list[str]:
    return sorted(m.uci() for m in board.legal_moves)


def _parse_move(board: chess.Board, text: str) -> chess.Move:
    """Accept UCI (e2e4, e7e8q) or SAN (Nf3, O-O) input."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty move")
    try:
        mv = chess.Move.from_uci(text)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass
    return board.parse_san(text)  # raises ValueError if unparseable / illegal


def _status(board: chess.Board, human_color: chess.Color) -> str:
    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        you = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(res)
        if you is None:
            verdict = "**Draw.**"
        elif you == human_color:
            verdict = "**You won!**"
        else:
            verdict = "**Agent wins.**"
        return f"{verdict} Final: `{res}`"
    to_move = "your" if board.turn == human_color else "agent's"
    check = " — **check!**" if board.is_check() else ""
    return f"{to_move} move{check}"


def _render(
    board: chess.Board,
    agent: GNNAgent,
    human_color: chess.Color,
    last_move: chess.Move | None,
    temperature: float,
    topk: int,
    show_arrows: bool,
) -> str:
    flipped = human_color == chess.BLACK
    if show_arrows and not board.is_game_over(claim_draw=True):
        ranking = agent.rank_moves(board, temperature=temperature)
        svg = render_prediction_svg(board, ranking, topk=int(topk))
        # render_prediction_svg doesn't take orientation/lastmove — redo
        # with arrows explicitly so we can pass those.
        k = min(int(topk), len(ranking.moves))
        max_p = max(ranking.probabilities[:k]) if k else 1.0
        arrows = []
        from chess_gnn.viz import _color_for_prob
        for m, p in zip(ranking.moves[:k], ranking.probabilities[:k]):
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square,
                                          color=_color_for_prob(p / (max_p or 1.0))))
        svg = chess.svg.board(
            board,
            arrows=arrows,
            lastmove=last_move,
            flipped=flipped,
            size=480,
        )
    else:
        svg = chess.svg.board(board, lastmove=last_move, flipped=flipped, size=480)
    return _wrap_svg(svg)


def build_app(agent: GNNAgent) -> gr.Blocks:
    with gr.Blocks(title="chess-gnn — play") as demo:
        gr.Markdown(
            "# chess-gnn — play against the engine\n"
            "Pick a color, choose a move from the dropdown (or type UCI/SAN), "
            "and the agent replies automatically."
        )

        # State: FEN string + color the human is playing.
        fen_state = gr.State(chess.STARTING_FEN)
        color_state = gr.State(chess.WHITE)  # True = white, False = black
        last_move_state = gr.State(None)     # chess.Move or None

        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML()
                status = gr.Markdown()
            with gr.Column(scale=1):
                color_choice = gr.Radio(
                    choices=["White", "Black"], value="White",
                    label="Play as",
                )
                temperature = gr.Slider(
                    0.01, 1.5, value=0.2, step=0.01,
                    label="Agent temperature (0 = argmax)",
                )
                show_arrows = gr.Checkbox(
                    value=False,
                    label="Show agent's predicted moves for the current position",
                )
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-k arrows")

                move_dropdown = gr.Dropdown(
                    choices=_legal_uci(chess.Board()),
                    label="Your move (legal UCI)",
                    interactive=True,
                )
                move_text = gr.Textbox(
                    label="…or type UCI / SAN (e.g. e2e4, Nf3, O-O)",
                    placeholder="e2e4",
                )
                with gr.Row():
                    play_btn = gr.Button("Play move", variant="primary")
                    new_btn = gr.Button("New game")
                    undo_btn = gr.Button("Undo")

        # ---------------------------------------------------------------- helpers
        def _agent_reply(board: chess.Board, temp: float) -> chess.Move | None:
            if board.is_game_over(claim_draw=True):
                return None
            return agent.select_move(board, temperature=float(temp))

        def _refresh(fen, human_color, last_move, temp, topk, show):
            board = chess.Board(fen)
            html = _render(board, agent, human_color, last_move, temp, topk, show)
            return (
                html,
                _status(board, human_color),
                gr.update(choices=_legal_uci(board), value=None),
            )

        # ---------------------------------------------------------------- actions
        def _new_game(color_label, temp, k, show):
            human_color = chess.WHITE if color_label == "White" else chess.BLACK
            board = chess.Board()
            last_move = None
            # If human chose black, agent (white) moves first.
            if human_color == chess.BLACK:
                mv = _agent_reply(board, temp)
                if mv is not None:
                    board.push(mv)
                    last_move = mv
            html = _render(board, agent, human_color, last_move, temp, k, show)
            return (
                board.fen(), human_color, last_move,
                html,
                _status(board, human_color),
                gr.update(choices=_legal_uci(board), value=None),
                "",  # clear move_text
            )

        def _play(fen, human_color, last_move, dropdown_val, text_val, temp, k, show):
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=True):
                html = _render(board, agent, human_color, last_move, temp, k, show)
                return (
                    fen, last_move, html, _status(board, human_color),
                    gr.update(choices=[], value=None), "",
                )
            if board.turn != human_color:
                # Shouldn't happen, but make it explicit instead of silently failing.
                html = _render(board, agent, human_color, last_move, temp, k, show)
                return (
                    fen, last_move, html,
                    "Not your turn — press 'New game' to restart.",
                    gr.update(choices=_legal_uci(board), value=None), "",
                )

            move_str = (text_val or "").strip() or (dropdown_val or "")
            if not move_str:
                html = _render(board, agent, human_color, last_move, temp, k, show)
                return (
                    fen, last_move, html,
                    "Pick a move from the dropdown or type one.",
                    gr.update(choices=_legal_uci(board), value=None), "",
                )
            try:
                human_move = _parse_move(board, move_str)
            except ValueError as e:
                html = _render(board, agent, human_color, last_move, temp, k, show)
                return (
                    fen, last_move, html,
                    f"**Illegal / unparseable move:** `{move_str}` ({e})",
                    gr.update(choices=_legal_uci(board), value=None), "",
                )

            board.push(human_move)
            new_last = human_move

            # Agent replies (unless the human move just ended the game).
            agent_move = _agent_reply(board, temp)
            if agent_move is not None:
                board.push(agent_move)
                new_last = agent_move

            html = _render(board, agent, human_color, new_last, temp, k, show)
            msg = _status(board, human_color)
            if agent_move is not None:
                msg = f"agent played **{agent_move.uci()}** — " + msg
            return (
                board.fen(), new_last, html, msg,
                gr.update(choices=_legal_uci(board), value=None),
                "",  # clear move_text
            )

        def _undo(fen, human_color, temp, k, show):
            board = chess.Board(fen)
            # Pop the agent's reply + the user's previous move so it's the
            # human's turn again.
            popped = 0
            for _ in range(2):
                if board.move_stack:
                    board.pop()
                    popped += 1
                if board.turn == human_color:
                    break
            last_move = board.peek() if board.move_stack else None
            html = _render(board, agent, human_color, last_move, temp, k, show)
            msg = _status(board, human_color)
            if popped == 0:
                msg = "Nothing to undo. " + msg
            return (
                board.fen(), last_move, html, msg,
                gr.update(choices=_legal_uci(board), value=None),
            )

        # ---------------------------------------------------------------- wiring
        play_btn.click(
            _play,
            [fen_state, color_state, last_move_state,
             move_dropdown, move_text, temperature, topk, show_arrows],
            [fen_state, last_move_state, board_html, status,
             move_dropdown, move_text],
        )
        new_btn.click(
            _new_game,
            [color_choice, temperature, topk, show_arrows],
            [fen_state, color_state, last_move_state,
             board_html, status, move_dropdown, move_text],
        )
        undo_btn.click(
            _undo,
            [fen_state, color_state, temperature, topk, show_arrows],
            [fen_state, last_move_state, board_html, status, move_dropdown],
        )
        # Re-render when arrows / top-k / temperature display toggles change.
        for ctrl in (show_arrows, topk, temperature):
            ctrl.change(
                _refresh,
                [fen_state, color_state, last_move_state, temperature, topk, show_arrows],
                [board_html, status, move_dropdown],
            )

        demo.load(
            _new_game,
            [color_choice, temperature, topk, show_arrows],
            [fen_state, color_state, last_move_state,
             board_html, status, move_dropdown, move_text],
        )

    return demo


def _load_agent(ckpt: Path, device: str) -> GNNAgent:
    if ckpt.exists():
        model = load_model(ckpt, device=device)
        print(f"loaded {ckpt} (config={model.config})")
    else:
        print(f"[warn] no checkpoint at {ckpt} — using a randomly-initialized model")
        model = ChessGNN()
        model.to(device)
    return GNNAgent(model, device=device, default_temperature=0.2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sl/sl_final.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agent = _load_agent(args.ckpt, args.device)
    demo = build_app(agent)
    demo.launch(share=args.share)

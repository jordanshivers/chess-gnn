"""Web app: play against a chess-gnn checkpoint with drag-and-drop.

Uses chessboard.js for the UI (drag/drop pieces) + chess.js for in-browser
legal-move validation; the Python backend (Flask) holds a single game state
and picks the agent's reply with the loaded model.

Run:
    pip install flask
    python -m app.web_play --ckpt checkpoints/rl/rl_final.pt
    # open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import chess
from flask import Flask, jsonify, render_template_string, request

from chess_gnn.model import ChessGNN, load_model
from chess_gnn.play import GNNAgent


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>chess-gnn — play</title>
  <link rel="stylesheet" href="/static/css/chessboard.min.css">
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; color: #222; }
    h1 { margin: 0 0 4px 0; }
    #subtitle { color: #666; margin-bottom: 16px; }
    #layout { display: flex; gap: 24px; align-items: flex-start; }
    #board { width: 480px; }
    #panel { min-width: 260px; }
    .row { margin: 10px 0; }
    label { font-size: 0.9em; color: #444; }
    input[type=range] { vertical-align: middle; }
    button { padding: 6px 14px; margin-right: 8px; cursor: pointer; }
    #status { margin-top: 12px; min-height: 1.4em; font-weight: 500; }
    #history { font-family: ui-monospace, monospace; font-size: 0.88em;
               background: #f6f6f6; padding: 8px; border-radius: 4px;
               max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
    .warn { color: #b00; }
  </style>
</head>
<body>
  <h1>chess-gnn</h1>
  <div id="subtitle">Drag a piece to move. The engine replies automatically.</div>

  <div id="layout">
    <div>
      <div id="board"></div>
    </div>
    <div id="panel">
      <div class="row">
        <label>Play as:
          <select id="color">
            <option value="white" selected>White</option>
            <option value="black">Black</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>Agent temperature: <span id="tempLabel">0.20</span></label><br>
        <input id="temp" type="range" min="0.01" max="1.5" step="0.01" value="0.2"
               style="width: 220px;">
      </div>
      <div class="row">
        <button id="newBtn">New game</button>
        <button id="undoBtn">Undo</button>
        <button id="flipBtn">Flip board</button>
      </div>
      <div id="status">your move.</div>
      <div class="row"><label>Move history</label></div>
      <div id="history"></div>
    </div>
  </div>

  <script src="/static/js/jquery.min.js"></script>
  <script src="/static/js/chess.min.js"></script>
  <script src="/static/js/chessboard.min.js"></script>

  <script>
  // Client-side chess.js instance mirrors the server state (we trust the
  // server's FEN on every round-trip).
  let game = new Chess();
  let humanColor = 'w';
  let board;

  function post(url, payload) {
    return fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload || {})
    }).then(r => r.json());
  }

  function setStatus(msg, warn) {
    const el = document.getElementById('status');
    el.innerHTML = msg;
    el.classList.toggle('warn', !!warn);
  }

  function renderHistory(sans) {
    const lines = [];
    for (let i = 0; i < sans.length; i += 2) {
      const num = (i / 2) + 1;
      const w = sans[i] || '';
      const b = sans[i + 1] || '';
      lines.push(`${num}. ${w} ${b}`);
    }
    document.getElementById('history').textContent = lines.join('\n');
  }

  function syncFromServer(resp) {
    game = new Chess(resp.fen);
    board.position(resp.fen);
    renderHistory(resp.san_history || []);
    if (resp.game_over) {
      setStatus(resp.result_text);
    } else {
      const whoseTurn = (game.turn() === humanColor) ? 'your' : 'agent\'s';
      const check = game.in_check() ? ' — <b>check!</b>' : '';
      setStatus(`${whoseTurn} move${check}`);
    }
  }

  function onDragStart(source, piece) {
    if (game.game_over()) return false;
    // Block dragging the opponent's or agent's pieces.
    if (game.turn() !== humanColor) return false;
    if (piece[0] !== humanColor) return false;
  }

  function onDrop(source, target) {
    // Build the move object. Auto-promote to queen; the server accepts the
    // promotion character if provided.
    const move = { from: source, to: target, promotion: 'q' };
    const legal = game.move(move);
    if (legal === null) return 'snapback';

    // Optimistically render; server confirms.
    board.position(game.fen());
    const uci = source + target + (legal.promotion ? legal.promotion : '');
    setStatus('thinking…');

    post('/move', { uci: uci }).then(resp => {
      if (resp.error) {
        setStatus('<b>server rejected move:</b> ' + resp.error, true);
        game.undo();
        board.position(game.fen());
        return;
      }
      syncFromServer(resp);
    });
  }

  function onSnapEnd() { board.position(game.fen()); }

  function newGame() {
    const colorSel = document.getElementById('color').value;
    humanColor = (colorSel === 'white') ? 'w' : 'b';
    post('/new', { color: colorSel }).then(resp => {
      board.orientation(colorSel);
      syncFromServer(resp);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    board = Chessboard('board', {
      draggable: true,
      position: 'start',
      pieceTheme: '/static/img/pieces/{piece}.png',
      onDragStart, onDrop, onSnapEnd,
    });
    document.getElementById('newBtn').addEventListener('click', newGame);
    document.getElementById('flipBtn').addEventListener('click', () => board.flip());
    document.getElementById('undoBtn').addEventListener('click', () => {
      post('/undo', {}).then(syncFromServer);
    });
    document.getElementById('temp').addEventListener('input', (e) => {
      document.getElementById('tempLabel').textContent = parseFloat(e.target.value).toFixed(2);
      post('/set_temperature', { temperature: parseFloat(e.target.value) });
    });
    newGame();
  });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Per-process game state. Serialized with a lock so concurrent requests from
# the browser can't interleave pushes on the board.
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, agent: GNNAgent):
        self.agent = agent
        self.board = chess.Board()
        self.human_color: chess.Color = chess.WHITE
        self.temperature: float = 0.2
        self.san_history: list[str] = []
        self.lock = threading.Lock()

    def _snapshot(self) -> dict:
        over = self.board.is_game_over(claim_draw=True)
        result_text = ""
        if over:
            res = self.board.result(claim_draw=True)
            winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(res)
            if winner is None:
                result_text = f"Draw ({res})."
            elif winner == self.human_color:
                result_text = f"You win! ({res})"
            else:
                result_text = f"Agent wins. ({res})"
        return {
            "fen": self.board.fen(),
            "san_history": list(self.san_history),
            "game_over": over,
            "result_text": result_text,
        }

    def _agent_reply_if_needed(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            return
        if self.board.turn == self.human_color:
            return
        mv = self.agent.select_move(self.board, temperature=float(self.temperature))
        self.san_history.append(self.board.san(mv))
        self.board.push(mv)

    def new_game(self, color: str) -> dict:
        with self.lock:
            self.board = chess.Board()
            self.human_color = chess.WHITE if color == "white" else chess.BLACK
            self.san_history = []
            self._agent_reply_if_needed()
            return self._snapshot()

    def play_move(self, uci: str) -> dict:
        with self.lock:
            if self.board.is_game_over(claim_draw=True):
                return {"error": "game is over"} | self._snapshot()
            if self.board.turn != self.human_color:
                return {"error": "not your turn"} | self._snapshot()
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError as e:
                return {"error": f"bad uci: {e}"} | self._snapshot()
            if mv not in self.board.legal_moves:
                # chessboard.js might send e.g. e7e8q when promotion is optional;
                # try without promotion as a last resort.
                fallback = chess.Move(mv.from_square, mv.to_square)
                if fallback in self.board.legal_moves:
                    mv = fallback
                else:
                    return {"error": "illegal move"} | self._snapshot()
            self.san_history.append(self.board.san(mv))
            self.board.push(mv)
            self._agent_reply_if_needed()
            return self._snapshot()

    def undo(self) -> dict:
        with self.lock:
            # Pop up to two plies so the board returns to the human's turn.
            for _ in range(2):
                if not self.board.move_stack:
                    break
                self.board.pop()
                if self.san_history:
                    self.san_history.pop()
                if self.board.turn == self.human_color:
                    break
            return self._snapshot()

    def set_temperature(self, t: float) -> None:
        self.temperature = max(0.01, min(1.5, float(t)))


# ---------------------------------------------------------------------------
# Flask plumbing.
# ---------------------------------------------------------------------------

def build_app(agent: GNNAgent) -> Flask:
    app = Flask(__name__)
    state = GameState(agent)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/new", methods=["POST"])
    def new_game():
        data = request.get_json(silent=True) or {}
        color = data.get("color", "white")
        return jsonify(state.new_game(color))

    @app.route("/move", methods=["POST"])
    def move():
        data = request.get_json(silent=True) or {}
        uci = (data.get("uci") or "").strip()
        if not uci:
            return jsonify({"error": "no uci"} | state._snapshot())
        return jsonify(state.play_move(uci))

    @app.route("/undo", methods=["POST"])
    def undo():
        return jsonify(state.undo())

    @app.route("/set_temperature", methods=["POST"])
    def set_temp():
        data = request.get_json(silent=True) or {}
        state.set_temperature(data.get("temperature", 0.2))
        return jsonify({"ok": True, "temperature": state.temperature})

    return app


def _load_agent(ckpt: Path, device: str, num_simulations: int) -> GNNAgent:
    if ckpt.exists():
        model = load_model(ckpt, device=device)
        print(f"loaded {ckpt} (config={model.config})")
    else:
        print(f"[warn] no checkpoint at {ckpt} — using a randomly-initialized model")
        model = ChessGNN()
        model.to(device)
    mode = f"MCTS ({num_simulations} sims)" if num_simulations > 0 else "raw policy"
    print(f"move selection: {mode}")
    return GNNAgent(
        model, device=device, default_temperature=0.2,
        num_simulations=num_simulations,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sl/sl_final.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--mcts-sims", type=int, default=0,
                   help="MCTS simulations per move (0 disables search).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agent = _load_agent(args.ckpt, args.device, args.mcts_sims)
    app = build_app(agent)
    print(f"→ open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

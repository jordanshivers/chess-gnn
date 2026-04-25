import random

import chess

from chess_gnn.moves import (
    NUM_MOVES,
    decode_move,
    is_promoting_pawn_move,
    legal_mask,
    move_to_index,
)


def _random_positions(n: int, max_plies: int = 60, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b)
    return out


def test_legal_mask_matches_python_chess():
    for b in _random_positions(50):
        mask = legal_mask(b)
        assert mask.shape == (NUM_MOVES,)
        legal_indices = {move_to_index(m) for m in b.legal_moves}
        set_indices = {int(i) for i in mask.nonzero().flatten().tolist()}
        assert set_indices == legal_indices


def test_decode_round_trip():
    for b in _random_positions(50):
        for mv in b.legal_moves:
            idx = move_to_index(mv)
            if is_promoting_pawn_move(b, mv):
                # Default queen-promotion path should decode to the queen promotion;
                # under-promotions reachable via choice 0/1/2.
                if mv.promotion == chess.QUEEN:
                    assert decode_move(b, idx) == mv
                else:
                    choice = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[mv.promotion]
                    assert decode_move(b, idx, underpromo_choice=choice) == mv
            else:
                assert decode_move(b, idx) == mv

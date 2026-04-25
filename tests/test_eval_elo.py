import math

from chess_gnn.eval_elo import (
    MatchResult,
    elo_diff_to_score,
    point_estimate,
    pooled_mle,
    score_to_elo_diff,
)


def test_score_elo_round_trip():
    for diff in [-400, -200, -50, 0, 50, 200, 400]:
        s = elo_diff_to_score(diff)
        back = score_to_elo_diff(s)
        assert abs(back - diff) < 1e-6


def test_point_estimate_50_percent():
    m = MatchResult(opponent_elo=1500, games=100, wins=50, draws=0, losses=50)
    r, ci = point_estimate(m)
    assert abs(r - 1500) < 1e-6
    assert ci > 0


def test_point_estimate_above_opponent():
    m = MatchResult(opponent_elo=1500, games=100, wins=75, draws=0, losses=25)
    r, _ = point_estimate(m)
    # 75% score ≈ +190 Elo above opponent
    assert 1670 < r < 1710


def test_pooled_mle_matches_single_opponent():
    # If there's only one opponent, pooled MLE should equal the point estimate.
    m = MatchResult(opponent_elo=1600, games=200, wins=120, draws=20, losses=60)
    pooled = pooled_mle([m])
    direct, _ = point_estimate(m)
    assert abs(pooled - direct) < 1.0


def test_pooled_mle_between_opponents():
    # Model beats 1500 at 80%, beats 2000 at 20% -> MLE should land in between.
    r1 = MatchResult(opponent_elo=1500, games=100, wins=80, draws=0, losses=20)
    r2 = MatchResult(opponent_elo=2000, games=100, wins=20, draws=0, losses=80)
    pooled = pooled_mle([r1, r2])
    assert 1500 < pooled < 2000

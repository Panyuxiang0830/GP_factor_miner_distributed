import pandas as pd
import pytest

from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.screening import screen_gp_factors, screen_raw_features


def _candidate(name, rank_ic):
    return Candidate(name, rank_ic, abs(rank_ic), 1.0, 0.0, 0, 1)


def test_selected_factor_retains_fitness_diagnostics():
    candidate = Candidate(
        "signal", 0.04, 0.04, 1.0, 0.0, 0, 1,
        fitness=0.035, segment_rank_ic_std=0.008, same_sign_ratio=1.0,
    )
    index = pd.Index(range(4), name="time_id")
    columns = pd.Index([0, 1, 2], name="asset_id")
    signal = pd.DataFrame([[1, 2, 3]] * 4, index=index, columns=columns, dtype=float)

    selected = screen_gp_factors(
        [candidate], {"signal": signal}, signal, threshold=0.03
    )

    assert selected[0].fitness == pytest.approx(0.035)
    assert selected[0].segment_rank_ic_std == pytest.approx(0.008)
    assert selected[0].same_sign_ratio == pytest.approx(1.0)


def test_raw_screen_requires_locked_bar_and_same_direction():
    evolution = pd.DataFrame(
        {"feature": ["pass", "wrong_sign", "too_low"], "rank_ic": [0.04, 0.04, 0.04]}
    )
    screening = pd.DataFrame(
        {"feature": ["pass", "wrong_sign", "too_low"], "rank_ic": [0.031, -0.05, 0.029]}
    )

    selected = screen_raw_features(evolution, screening, threshold=0.03)

    assert selected["feature"].tolist() == ["pass"]


def test_gp_screen_requires_same_direction_and_point_zero_four():
    index = pd.Index([0, 1], name="time_id")
    data = {
        "pass": pd.DataFrame([[1, 2, 3], [3, 2, 1]], index=index),
        "wrong_sign": pd.DataFrame([[3, 2, 1], [1, 2, 3]], index=index),
        "too_low": pd.DataFrame([[1, 3, 2], [2, 1, 3]], index=index),
    }
    target = data["pass"]
    candidates = [_candidate("pass", 0.05), _candidate("wrong_sign", 0.05), _candidate("too_low", 0.05)]

    selected = screen_gp_factors(candidates, data, target, threshold=0.04, corr_threshold=0.9, limit=10)

    assert [item.name for item in selected] == ["pass"]


def test_correlated_factor_keeps_higher_screening_rank_ic():
    index = pd.Index([0, 1], name="time_id")
    target = pd.DataFrame([[1, 2, 3], [3, 2, 1]], index=index)
    data = {
        "a": target.copy(),
        "b": target * 2,
    }
    candidates = [_candidate("a", 0.06), _candidate("b", 0.05)]

    selected = screen_gp_factors(candidates, data, target, threshold=0.04, corr_threshold=0.9, limit=10)

    assert [item.name for item in selected] == ["a"]

import numpy as np
import pandas as pd

from gp_factor_miner.operators import OPERATORS


def test_protected_division_returns_nan_for_zero_denominator():
    left = pd.DataFrame([[1.0, 2.0]])
    right = pd.DataFrame([[0.0, 2.0]])

    result = OPERATORS["/"](left, right)

    assert np.isnan(result.iloc[0, 0])
    assert result.iloc[0, 1] == 1.0


def test_rank_is_cross_sectional():
    values = pd.DataFrame([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])

    result = OPERATORS["rank"](values)

    assert result.iloc[0].tolist() == [1.0, 1 / 3, 2 / 3]


def test_rolling_mean_is_backward_looking():
    values = pd.DataFrame([[1.0], [3.0], [100.0]])

    result = OPERATORS["mean"](values, 2)

    assert result.iloc[:2, 0].tolist() == [1.0, 2.0]


def test_ema_uses_backward_looking_exponential_weights():
    values = pd.DataFrame({"asset": [1.0, 2.0, 3.0]})

    result = OPERATORS["ema"](values, 2)

    expected = values.ewm(span=2, adjust=False, min_periods=1).mean()
    pd.testing.assert_frame_equal(result, expected)

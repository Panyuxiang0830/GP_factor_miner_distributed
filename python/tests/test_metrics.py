import numpy as np
import pandas as pd
import pytest

from gp_factor_miner.metrics import rank_ic


def panel(values):
    return pd.DataFrame(values, index=pd.Index(range(len(values)), name="time_id"))


def test_rank_ic_is_one_for_matching_cross_sectional_order():
    factor = panel([[1, 2, 3], [3, 2, 1]])
    target = panel([[10, 20, 30], [30, 20, 10]])

    result = rank_ic(factor, target)

    assert result.mean == pytest.approx(1.0)
    assert result.valid_times == 2
    assert result.valid_ratio == pytest.approx(1.0)


def test_rank_ic_preserves_negative_direction():
    factor = panel([[1, 2, 3], [1, 2, 3]])
    target = panel([[30, 20, 10], [30, 20, 10]])

    assert rank_ic(factor, target).mean == pytest.approx(-1.0)


def test_rank_ic_rejects_constant_cross_sections():
    factor = panel([[1, 1, 1], [1, 2, 3]])
    target = panel([[1, 2, 3], [1, 2, 3]])

    result = rank_ic(factor, target)

    assert result.valid_times == 1
    assert result.valid_ratio == pytest.approx(0.5)
    assert np.isfinite(result.mean)


def test_missing_ratio_uses_only_target_eligible_cells():
    factor = panel([[1.0, 2.0, np.nan], [3.0, np.nan, 1.0]])
    target = panel([[1.0, 2.0, np.nan], [3.0, np.nan, 1.0]])

    result = rank_ic(factor, target)

    assert result.missing_ratio == pytest.approx(0.0)

import pandas as pd
import pytest

from gp_factor_miner.raw_features import (
    feature_sampling_probabilities,
    lock_raw_threshold,
    profile_raw_features,
)


def test_sampling_keeps_nonzero_probability_for_every_feature():
    profile = pd.DataFrame(
        {"feature": ["a", "b"], "rank_ic": [0.2, 0.0]}
    )

    probabilities = feature_sampling_probabilities(profile, uniform_share=0.5)

    assert probabilities[1] > 0
    assert probabilities.sum() == pytest.approx(1.0)


def test_threshold_locks_before_screening_from_evolution_profile():
    strong = pd.DataFrame({"rank_ic": [0.031] * 30 + [0.0]})
    weak = pd.DataFrame({"rank_ic": [0.031] * 29 + [0.0]})

    assert lock_raw_threshold(strong) == pytest.approx(0.03)
    assert lock_raw_threshold(weak) == pytest.approx(0.01)


def test_profile_raw_features_uses_cross_sectional_rank_ic():
    frame = pd.DataFrame(
        {
            "time_id": [0, 0, 0, 1, 1, 1],
            "asset_id": [0, 1, 2, 0, 1, 2],
            "feature_000": [1, 2, 3, 3, 2, 1],
            "target": [10, 20, 30, 30, 20, 10],
        }
    )

    profile = profile_raw_features(frame, ["feature_000"])

    assert profile.loc[0, "feature"] == "feature_000"
    assert profile.loc[0, "rank_ic"] == pytest.approx(1.0)

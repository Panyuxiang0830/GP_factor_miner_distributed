import pandas as pd

from gp_factor_miner.generator import FactorGenerator


def test_reference_tuple_protocol_computes_factor():
    generator = FactorGenerator(
        {"feature_001": pd.DataFrame([[1, 2], [3, 5]], dtype=float)}
    )
    generator.register_factor("f1", ("delta", "feature_001", 1))

    actual = generator.compute_factor("f1")

    assert actual.iloc[1].tolist() == [2, 3]


def test_batch_compute_preserves_requested_names():
    generator = FactorGenerator(
        {"feature_001": pd.DataFrame([[1, 2], [3, 5]], dtype=float)}
    )
    generator.register_factor("ranked", ("rank", "feature_001"))
    generator.register_factor("lagged", ("lag", "feature_001", 1))

    result = generator.batch_compute(["lagged", "ranked"], max_workers=1)

    assert list(result) == ["lagged", "ranked"]

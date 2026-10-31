import pandas as pd
from pandas.testing import assert_frame_equal

from gp_factor_miner.expressions import (
    evaluate_expression,
    expression_depth,
    expression_nodes,
    normalize_expression,
)


def test_future_change_does_not_change_past_factor_values():
    original = pd.DataFrame([[1, 2], [3, 4], [5, 6]], dtype=float)
    changed = original.copy()
    changed.iloc[2] = [500, 600]
    expression = ("mean", "feature_001", 2)

    assert_frame_equal(
        evaluate_expression(expression, {"feature_001": original}).iloc[:2],
        evaluate_expression(expression, {"feature_001": changed}).iloc[:2],
    )


def test_commutative_expression_normalizes_operand_order():
    assert normalize_expression(("+", "feature_002", "feature_001")) == (
        "+",
        "feature_001",
        "feature_002",
    )


def test_expression_shape_metrics():
    expression = ("rank", ("delta", "feature_001", 2))
    assert expression_depth(expression) == 2
    assert expression_nodes(expression) == 3

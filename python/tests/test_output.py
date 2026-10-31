import json

import pandas as pd

from gp_factor_miner.output import write_outputs


def test_write_outputs_creates_five_artifacts(tmp_path):
    paths = write_outputs(
        tmp_path,
        raw_features=[{"feature": "feature_001", "evolution_rank_ic": 0.04, "screening_rank_ic": 0.03}],
        gp_factors=[{"name": "gp_0001", "expression": ("rank", "feature_001")}],
        metrics=pd.DataFrame([{"name": "gp_0001", "screening_rank_ic": 0.05}]),
        history=pd.DataFrame([{"generation": 0, "best_abs_rank_ic": 0.05}]),
        run_config={"screening_end": 4, "sealed_start": 5},
    )

    assert len(paths) == 5
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    payload = json.loads((tmp_path / "selected_gp_factors.json").read_text())
    assert payload[0]["expression"] == ["rank", "feature_001"]

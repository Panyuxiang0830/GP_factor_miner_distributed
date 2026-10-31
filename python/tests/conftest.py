from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_panel(path: Path, time_ids=range(10)) -> Path:
    rows = []
    for time_id in time_ids:
        for asset_id in (0, 1, 2):
            rows.append(
                {
                    "time_id": time_id,
                    "asset_id": asset_id,
                    "feature_000": float(time_id + asset_id),
                    "feature_001": float(time_id - asset_id),
                    "target": float(asset_id - 1),
                }
            )
    output = path / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output)
    return output


def write_eod_pickles(path: Path, time_ids=range(40)) -> tuple[Path, Path]:
    n_dates = len(tuple(time_ids))
    evolution_count = max(1, int(n_dates * 0.4))
    screening_count = max(1, int(n_dates * 0.1))
    reserved_count = n_dates - evolution_count - screening_count
    dates = pd.Index(
        [
            *pd.bdate_range("2016-01-04", periods=evolution_count).strftime("%Y%m%d"),
            *pd.bdate_range("2022-01-04", periods=screening_count).strftime("%Y%m%d"),
            *pd.bdate_range("2023-01-03", periods=reserved_count).strftime("%Y%m%d"),
        ]
    )
    assets = pd.Index(
        ["000001", "000002", "000003", "600000", "600001", "600002", "830001", "830002"],
        name="Stkcd",
    )
    base = np.arange(len(dates), dtype=float)[:, None]
    offsets = np.arange(len(assets), dtype=float)[None, :]
    close = pd.DataFrame(base + offsets + 10.0, index=dates, columns=assets)
    rng = np.random.default_rng(2026)
    payload = {
        "close": close,
        "amount": close * (offsets + 1.0),
        "trade_status": pd.DataFrame(True, index=dates, columns=assets),
        "ST": pd.DataFrame(False, index=dates, columns=assets),
    }
    for number in range(12):
        payload[f"signal_{number:02d}"] = pd.DataFrame(
            rng.normal(size=close.shape) + number * offsets / 10.0,
            index=dates,
            columns=assets,
        )
    data_file = path / "eod"
    label_file = path / "label_10.pkl"
    pd.to_pickle(payload, data_file)
    label = pd.DataFrame(
        rng.normal(size=close.shape),
        index=dates,
        columns=[
            "000001.SZ",
            "000002.SZ",
            "000003.SZ",
            "600000.SH",
            "600001.SH",
            "600002.SH",
            "830001.BJ",
            "830002.BJ",
        ],
    )
    pd.to_pickle(label, label_file)
    return data_file, label_file

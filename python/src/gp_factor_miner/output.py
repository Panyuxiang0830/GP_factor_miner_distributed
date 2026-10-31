from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_json(path, payload) -> Path:
    """Atomically write a JSON artifact, creating its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, payload)
    return destination


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_outputs(
    output_dir,
    raw_features,
    gp_factors,
    metrics,
    history,
    run_config,
):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        destination / "selected_raw_features.json",
        destination / "selected_gp_factors.json",
        destination / "factor_metrics.csv",
        destination / "gp_history.csv",
        destination / "run_config.json",
    ]
    atomic_json(paths[0], raw_features)
    atomic_json(paths[1], gp_factors)
    _atomic_csv(paths[2], metrics)
    _atomic_csv(paths[3], history)
    atomic_json(paths[4], run_config)
    return paths

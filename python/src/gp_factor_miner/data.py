from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


@dataclass(frozen=True)
class DataSchema:
    feature_names: tuple[str, ...]
    required_names: tuple[str, ...] = ("time_id", "asset_id", "target")


@dataclass(frozen=True)
class TimeSplit:
    evolution_end: object
    screening_end: object
    n_time_ids: int
    evolution_time_ids: tuple[object, ...]
    screening_time_ids: tuple[object, ...]


@dataclass(frozen=True)
class EODPanelDataset:
    """Aligned wide EOD feature panels and their forward-return target."""

    features: Mapping[str, pd.DataFrame]
    target: pd.DataFrame
    split: TimeSplit
    source_field_count: int
    excluded_fields: tuple[str, ...]
    common_date_count: int
    common_asset_count: int

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features)


DEFAULT_EOD_EXCLUDED_FIELDS = ("trade_status", "ST")
EOD_EVOLUTION_START = pd.Timestamp("2016-01-01")
EOD_EVOLUTION_END = pd.Timestamp("2021-12-31")
EOD_SCREENING_START = pd.Timestamp("2022-01-01")
EOD_SCREENING_END = pd.Timestamp("2022-12-31")


def _dataset(data_dir: str | Path) -> ds.Dataset:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"data directory does not exist: {path}")
    return ds.dataset(path, format="parquet", exclude_invalid_files=True)


def discover_schema(data_dir: str | Path) -> DataSchema:
    names = _dataset(data_dir).schema.names
    required = ("time_id", "asset_id", "target")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    features = tuple(sorted(name for name in names if name.startswith("feature_")))
    if not features:
        raise ValueError("no feature_* columns found")
    return DataSchema(feature_names=features)


def compute_time_split(data_dir: str | Path) -> TimeSplit:
    table = _dataset(data_dir).to_table(columns=["time_id"])
    time_ids = tuple(sorted(set(table.column("time_id").to_pylist())))
    return compute_time_split_from_ids(time_ids)


def compute_time_split_from_ids(time_ids: Iterable[object]) -> TimeSplit:
    time_ids = tuple(sorted(set(time_ids)))
    if len(time_ids) < 10:
        raise ValueError("at least 10 distinct time_id values are required")
    evolution_count = max(1, int(len(time_ids) * 0.4))
    research_count = max(evolution_count + 1, int(len(time_ids) * 0.5))
    evolution = time_ids[:evolution_count]
    screening = time_ids[evolution_count:research_count]
    return TimeSplit(
        evolution_end=evolution[-1],
        screening_end=screening[-1],
        n_time_ids=len(time_ids),
        evolution_time_ids=evolution,
        screening_time_ids=screening,
    )


def compute_eod_time_split_from_ids(time_ids: Iterable[object]) -> TimeSplit:
    """Use fixed calendar windows for EOD evolution and OOS screening."""

    normalized_ids = pd.Index(str(value) for value in time_ids).unique()
    parsed_dates = pd.to_datetime(normalized_ids, errors="coerce")
    if parsed_dates.isna().any():
        invalid = normalized_ids[parsed_dates.isna()].tolist()[:3]
        raise ValueError(f"EOD time ids contain invalid dates: {invalid}")

    order = np.argsort(parsed_dates.values, kind="stable")
    normalized_ids = normalized_ids.take(order)
    parsed_dates = parsed_dates.take(order)
    evolution = tuple(
        normalized_ids[
            (parsed_dates >= EOD_EVOLUTION_START)
            & (parsed_dates <= EOD_EVOLUTION_END)
        ]
    )
    screening = tuple(
        normalized_ids[
            (parsed_dates >= EOD_SCREENING_START)
            & (parsed_dates <= EOD_SCREENING_END)
        ]
    )
    if not evolution:
        raise ValueError("EOD data have no dates in the 2016-01-01 to 2021-12-31 evolution window")
    if not screening:
        raise ValueError("EOD data have no dates in the 2022-01-01 to 2022-12-31 screening window")
    return TimeSplit(
        evolution_end=evolution[-1],
        screening_end=screening[-1],
        n_time_ids=len(normalized_ids),
        evolution_time_ids=evolution,
        screening_time_ids=screening,
    )


def _normalize_axis(values: Iterable[object]) -> pd.Index:
    return pd.Index((str(value) for value in values))


def _normalize_label_columns(values: Iterable[object]) -> pd.Index:
    normalized = pd.Index(str(value).split(".", 1)[0] for value in values)
    if not normalized.is_unique:
        raise ValueError("label asset codes are not unique after exchange suffix removal")
    return normalized


def load_eod_panel_dataset(
    data_file: str | Path,
    label_file: str | Path,
    *,
    excluded_fields: Iterable[str] = DEFAULT_EOD_EXCLUDED_FIELDS,
    dtype: str = "float32",
) -> EODPanelDataset:
    """Load the native ``field -> date x asset`` EOD pickle without long reshaping.

    The target file uses exchange-suffixed codes (for example ``000001.SZ``),
    while the EOD panels use bare six-digit codes.  Assets and dates are
    intersected explicitly, and the target is masked to tradable, non-ST rows.
    """

    data_path = Path(data_file)
    label_path = Path(label_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"EOD data file does not exist: {data_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"label file does not exist: {label_path}")

    payload = pd.read_pickle(data_path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("EOD pickle must contain a non-empty field-to-DataFrame dict")
    if "close" not in payload or not isinstance(payload["close"], pd.DataFrame):
        raise ValueError("EOD pickle must contain a DataFrame named 'close'")

    reference = payload["close"].copy(deep=False)
    reference.index = _normalize_axis(reference.index)
    reference.columns = _normalize_axis(reference.columns)
    if not reference.index.is_unique or not reference.columns.is_unique:
        raise ValueError("EOD close panel must have unique date and asset axes")

    target = pd.read_pickle(label_path)
    if not isinstance(target, pd.DataFrame):
        raise ValueError("label pickle must contain a pandas DataFrame")
    target = target.copy(deep=False)
    target.index = _normalize_axis(target.index)
    target.columns = _normalize_label_columns(target.columns)

    common_dates = reference.index.intersection(target.index, sort=False)
    common_assets = reference.columns.intersection(target.columns, sort=False)
    if len(common_dates) < 10:
        raise ValueError("EOD data and label have fewer than 10 common dates")
    if len(common_assets) < 3:
        raise ValueError("EOD data and label have fewer than 3 common assets")
    split = compute_eod_time_split_from_ids(common_dates)
    research_dates = pd.Index(
        (*split.evolution_time_ids, *split.screening_time_ids)
    )

    excluded = set(excluded_fields)
    features: dict[str, pd.DataFrame] = {}
    for name, value in payload.items():
        if name in excluded or not isinstance(value, pd.DataFrame):
            continue
        if not all(np.issubdtype(item, np.number) for item in value.dtypes):
            continue
        panel = value.copy(deep=False)
        panel.index = _normalize_axis(panel.index)
        panel.columns = _normalize_axis(panel.columns)
        panel = panel.reindex(index=research_dates, columns=common_assets)
        features[str(name)] = panel.astype(dtype)
    if not features:
        raise ValueError("EOD pickle contains no usable numeric feature panels")

    target = target.reindex(index=research_dates, columns=common_assets).astype(dtype)
    if "trade_status" in payload:
        trade_status = payload["trade_status"].copy(deep=False)
        trade_status.index = _normalize_axis(trade_status.index)
        trade_status.columns = _normalize_axis(trade_status.columns)
        target = target.where(
            trade_status.reindex(index=research_dates, columns=common_assets).fillna(False)
        )
    if "ST" in payload:
        st_status = payload["ST"].copy(deep=False)
        st_status.index = _normalize_axis(st_status.index)
        st_status.columns = _normalize_axis(st_status.columns)
        target = target.where(
            ~st_status.reindex(index=research_dates, columns=common_assets).fillna(True)
        )

    return EODPanelDataset(
        features=features,
        target=target,
        split=split,
        source_field_count=len(payload),
        excluded_fields=tuple(sorted(excluded)),
        common_date_count=len(common_dates),
        common_asset_count=len(common_assets),
    )


def split_eod_panels(
    dataset: EODPanelDataset,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    evolution_ids = list(dataset.split.evolution_time_ids)
    screening_ids = list(dataset.split.screening_time_ids)
    evolution = {
        name: panel.loc[evolution_ids] for name, panel in dataset.features.items()
    }
    screening = {
        name: panel.loc[screening_ids] for name, panel in dataset.features.items()
    }
    return (
        evolution,
        dataset.target.loc[evolution_ids],
        screening,
        dataset.target.loc[screening_ids],
    )


def load_research_rows(
    data_dir: str | Path,
    split: TimeSplit,
    columns: Iterable[str],
) -> pd.DataFrame:
    requested = list(columns)
    dataset = _dataset(data_dir)
    unknown = [name for name in requested if name not in dataset.schema.names]
    if unknown:
        raise ValueError(f"unknown requested columns: {unknown}")
    table = dataset.to_table(
        columns=requested,
        filter=ds.field("time_id") <= split.screening_end,
    )
    return table.to_pandas().sort_values(
        [name for name in ("time_id", "asset_id") if name in requested],
        kind="stable",
    ).reset_index(drop=True)


def split_research_frame(
    frame: pd.DataFrame,
    split: TimeSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evolution = frame[frame["time_id"] <= split.evolution_end].copy()
    screening = frame[
        (frame["time_id"] > split.evolution_end)
        & (frame["time_id"] <= split.screening_end)
    ].copy()
    return evolution, screening

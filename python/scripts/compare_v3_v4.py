"""Compare persisted V3 and V4 GP-factor run evidence.

The script reads only run artifacts. It never treats an absent or unreadable
artifact as an empty result, and exposes unsupported evidence as ``null``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys
from typing import Any


_METRIC_FIELDS = (
    "qualified_count",
    "selected_count",
    "survival_rate",
    "median_validation_rankic",
    "median_validation_abs_rankic",
    "top5_feature_share",
    "validity_rate",
    "runtime_seconds",
    "peak_memory_mb",
)

_LEGACY_FACTOR_IDENTITY_COLUMNS = frozenset(
    {"name", "factor", "factor_name", "expression"}
)
_LEGACY_FACTOR_QUALITY_COLUMNS = frozenset(
    {
        "rank_ic",
        "abs_rank_ic",
        "evolution_rank_ic",
        "screening_rank_ic",
        "abs_screening_rank_ic",
        "fitness",
    }
)


@dataclass(frozen=True)
class _CsvArtifact:
    rows: list[dict[str, str]]
    fields: frozenset[str]


def _read_csv(path: Path) -> _CsvArtifact | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None
            return _CsvArtifact(
                rows=list(reader),
                fields=frozenset(reader.fieldnames),
            )
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _valid_threshold(value: Any) -> float | None:
    threshold = _finite_float(value)
    return threshold if threshold is not None and threshold >= 0 else None


def _threshold_from_mapping(mapping: dict[str, Any]) -> float | None:
    for field in ("final_rankic_threshold", "gp_threshold", "rankic_threshold"):
        if field in mapping and (threshold := _valid_threshold(mapping[field])) is not None:
            return threshold
    return None


def _rankic_threshold(
    config: dict[str, Any] | None,
    diagnostics: tuple[_CsvArtifact | None, ...],
) -> float | None:
    if isinstance(config, dict):
        if (threshold := _threshold_from_mapping(config)) is not None:
            return threshold
        nested = config.get("gp_config")
        if isinstance(nested, dict) and (threshold := _threshold_from_mapping(nested)) is not None:
            return threshold
    observed = {
        threshold
        for artifact in diagnostics
        if artifact is not None
        for row in artifact.rows
        if (threshold := _threshold_from_mapping(row)) is not None
    }
    return observed.pop() if len(observed) == 1 else None


def _qualified_rows(
    validation: _CsvArtifact | None,
    decisions: _CsvArtifact | None,
    threshold: float | None,
) -> list[dict[str, str]] | None:
    source = validation if validation is not None else decisions
    if source is None or threshold is None:
        return None
    if "screening_rank_ic" not in source.fields:
        return None
    return [
        row
        for row in source.rows
        if (rankic := _finite_float(row.get("screening_rank_ic"))) is not None
        and abs(rankic) >= threshold
        and _truthy(row.get("same_direction"))
    ]


def _selected_factors(output_dir: Path) -> list[dict[str, Any]] | None:
    payload = _read_json(output_dir / "selected_gp_factors.json")
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    return None


def _is_legacy_factor_metrics(artifact: _CsvArtifact | None) -> bool:
    """Recognize only V3 selected-factor metrics, not arbitrary CSV rows."""
    return artifact is not None and bool(
        artifact.fields & _LEGACY_FACTOR_IDENTITY_COLUMNS
    ) and bool(artifact.fields & _LEGACY_FACTOR_QUALITY_COLUMNS)


def _feature_mentions(expression: Any):
    if isinstance(expression, str):
        try:
            decoded = json.loads(expression)
        except json.JSONDecodeError:
            decoded = expression
        if decoded != expression:
            yield from _feature_mentions(decoded)
        elif expression.startswith("feature_"):
            yield expression
        return
    if isinstance(expression, (list, tuple)):
        for item in expression:
            yield from _feature_mentions(item)


def _top5_feature_share(factors: list[dict[str, Any]] | None) -> float | None:
    if factors is None:
        return None
    counts: Counter[str] = Counter()
    for factor in factors:
        if isinstance(factor, dict):
            counts.update(_feature_mentions(factor.get("expression")))
    total = sum(counts.values())
    if not total:
        return None
    top_five = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return sum(count for _, count in top_five) / total


def _validation_rankics(rows: list[dict[str, Any]]) -> list[float]:
    return [
        rankic
        for row in rows
        if (rankic := _finite_float(row.get("screening_rank_ic"))) is not None
    ]


def _runtime_seconds(history: _CsvArtifact | None) -> float | None:
    """Read only explicitly seconds-denominated duration fields.

    Per-row ``runtime_seconds`` and ``duration_seconds`` are summed, while
    cumulative ``elapsed_seconds`` uses its maximum observation.
    """
    if history is None:
        return None
    for field in ("runtime_seconds", "duration_seconds", "elapsed_seconds"):
        if field not in history.fields:
            continue
        values = [
            value
            for row in history.rows
            if (value := _finite_float(row.get(field))) is not None and value >= 0
        ]
        if values:
            runtime = sum(values) if field != "elapsed_seconds" else max(values)
            return runtime if math.isfinite(runtime) else None
    return None


def _memory_values_mb(payload: Any) -> list[float]:
    values = []
    if isinstance(payload, dict):
        for field in ("peak_memory_mb", "peak_rss_mb", "max_rss_mb"):
            if (value := _finite_float(payload.get(field))) is not None and value >= 0:
                values.append(value)
        for field in ("peak_memory_bytes", "peak_rss_bytes", "max_rss_bytes"):
            if (value := _finite_float(payload.get(field))) is not None and value >= 0:
                values.append(value / (1024 * 1024))
        for value in payload.values():
            values.extend(_memory_values_mb(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_memory_values_mb(value))
    return values


def _peak_memory_mb(output_dir: Path) -> float | None:
    values = []
    for name in ("resource_summary.json", "resources.json"):
        values.extend(_memory_values_mb(_read_json(output_dir / name)))
    for name in ("resource_summary.csv", "resources.csv"):
        artifact = _read_csv(output_dir / name)
        if artifact is not None:
            values.extend(_memory_values_mb(artifact.rows))
    return max(values) if values else None


def summarize_run(output_dir: str | Path) -> dict[str, int | float | None]:
    """Return a comparable evidence row for a persisted V3 or V4 run."""
    path = Path(output_dir)
    config_payload = _read_json(path / "run_config.json")
    config = config_payload if isinstance(config_payload, dict) else None
    validation = _read_csv(path / "validation_rankic.csv")
    decisions = _read_csv(path / "correlation_decisions.csv")
    factor_metrics = _read_csv(path / "factor_metrics.csv")
    factors = _selected_factors(path)
    qualified_rows = _qualified_rows(
        validation, decisions, _rankic_threshold(config, (validation, decisions))
    )

    if factors is not None:
        selected_count = len(factors)
    elif decisions is not None and "selected" in decisions.fields:
        selected_count = sum(_truthy(row.get("selected")) for row in decisions.rows)
    elif _is_legacy_factor_metrics(factor_metrics):
        selected_count = len(factor_metrics.rows)
    else:
        selected_count = None

    rankics = _validation_rankics(qualified_rows or [])
    if validation is None or not validation.rows or "screening_rank_ic" not in validation.fields:
        validity_rate = None
    else:
        valid_rows = [
            row
            for row in validation.rows
            if _finite_float(row.get("screening_rank_ic")) is not None
            and not str(row.get("error", "")).strip()
        ]
        validity_rate = len(valid_rows) / len(validation.rows)
    qualified_count = len(qualified_rows) if qualified_rows is not None else None
    return {
        "qualified_count": qualified_count,
        "selected_count": selected_count,
        "survival_rate": (
            selected_count / qualified_count
            if selected_count is not None and qualified_count not in (None, 0)
            else None
        ),
        "median_validation_rankic": median(rankics) if rankics else None,
        "median_validation_abs_rankic": (
            median([abs(value) for value in rankics]) if rankics else None
        ),
        "top5_feature_share": _top5_feature_share(factors),
        "validity_rate": validity_rate,
        "runtime_seconds": _runtime_seconds(_read_csv(path / "gp_history.csv")),
        "peak_memory_mb": _peak_memory_mb(path),
    }


def compare_runs(v3_output: str | Path, v4_output: str | Path) -> dict[str, dict]:
    """Return evidence rows for the named V3 and V4 output directories."""
    return {"v3": summarize_run(v3_output), "v4": summarize_run(v4_output)}


def _write_table(report: dict[str, dict]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=("run", *_METRIC_FIELDS))
    writer.writeheader()
    for run in ("v3", "v4"):
        writer.writerow({"run": run, **report[run]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare V3 and V4 GP run outputs")
    parser.add_argument("--v3-output", type=Path, required=True)
    parser.add_argument("--v4-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.v3_output, args.v4_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    _write_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

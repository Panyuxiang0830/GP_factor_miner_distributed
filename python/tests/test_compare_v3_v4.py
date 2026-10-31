import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_v3_v4 import compare_runs, main, summarize_run


def write_fake_run(path, *, qualified, selected):
    path.mkdir()
    validation_rows = [
        {
            "expression": f'["rank","feature_{index % 6:03d}"]',
            "evolution_rank_ic": "0.04",
            "screening_rank_ic": "0.05",
            "abs_screening_rank_ic": "0.05",
            "same_direction": "1",
            "valid_ratio": "0.90",
            "error": "",
        }
        for index in range(qualified)
    ]
    with (path / "validation_rankic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_rows[0])
        writer.writeheader()
        writer.writerows(validation_rows)

    decisions = [
        {**row, "selected": int(index < selected)}
        for index, row in enumerate(validation_rows)
    ]
    with (path / "correlation_decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0])
        writer.writeheader()
        writer.writerows(decisions)

    factors = [
        {
            "expression": ["+", "feature_000", ["rank", "feature_001"]],
            "screening_rank_ic": 0.05,
        }
        for _ in range(selected)
    ]
    (path / "selected_gp_factors.json").write_text(json.dumps(factors))
    with (path / "gp_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["runtime_seconds"])
        writer.writeheader()
        writer.writerow({"runtime_seconds": "12.5"})
    (path / "resource_summary.json").write_text(json.dumps({"peak_memory_mb": 456.0}))
    (path / "run_config.json").write_text(json.dumps({"final_rankic_threshold": 0.03}))


def test_compare_reports_required_acceptance_metrics(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=100, selected=20)
    write_fake_run(tmp_path / "v4", qualified=120, selected=36)

    report = compare_runs(tmp_path / "v3", tmp_path / "v4")

    assert report["v3"]["survival_rate"] == pytest.approx(0.20)
    assert report["v4"]["survival_rate"] == pytest.approx(0.30)
    assert report["v4"]["qualified_count"] == 120
    assert report["v4"]["selected_count"] == 36
    assert report["v4"]["median_validation_abs_rankic"] == pytest.approx(0.05)
    assert report["v4"]["top5_feature_share"] == pytest.approx(1.0)
    assert report["v4"]["validity_rate"] == pytest.approx(1.0)
    assert report["v4"]["runtime_seconds"] == pytest.approx(12.5)
    assert report["v4"]["peak_memory_mb"] == pytest.approx(456.0)


def test_compare_keeps_zero_qualified_when_validation_diagnostics_exist(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    validation_path = tmp_path / "v4" / "validation_rankic.csv"
    rows = list(csv.DictReader(validation_path.open()))
    rows[0]["screening_rank_ic"] = "0.02"
    rows[0]["abs_screening_rank_ic"] = "0.02"
    with validation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    report = compare_runs(tmp_path / "v3", tmp_path / "v4")

    assert report["v4"]["qualified_count"] == 0
    assert report["v4"]["survival_rate"] is None


def test_missing_artifacts_report_unknown_not_empty(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()

    report = summarize_run(missing)

    assert all(value is None for value in report.values())


def test_header_only_diagnostics_are_a_known_empty_result(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_config.json").write_text(json.dumps({"final_rankic_threshold": 0.03}))
    for name, fields in (
        ("validation_rankic.csv", ["screening_rank_ic", "same_direction", "error"]),
        ("correlation_decisions.csv", ["selected"]),
    ):
        with (output / name).open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    (output / "selected_gp_factors.json").write_text("[]")

    report = summarize_run(output)

    assert report["qualified_count"] == 0
    assert report["selected_count"] == 0
    assert report["survival_rate"] is None
    assert report["median_validation_rankic"] is None
    assert report["validity_rate"] is None


def test_feature_mentions_decode_nested_json_strings_deterministically(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    (tmp_path / "v4" / "selected_gp_factors.json").write_text(
        json.dumps(
            [
                {
                    "expression": json.dumps(
                        [
                            "+",
                            json.dumps(["+", "feature_000", "feature_001"]),
                            "feature_002",
                            "feature_003",
                            "feature_004",
                            "feature_005",
                            "feature_006",
                        ]
                    )
                },
                {"expression": "[not valid json"},
            ]
        )
    )

    first = summarize_run(tmp_path / "v4")
    second = summarize_run(tmp_path / "v4")

    assert first["top5_feature_share"] == pytest.approx(5 / 7)
    assert second["top5_feature_share"] == first["top5_feature_share"]


def test_qualification_requires_persisted_threshold_and_true_direction(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    output = tmp_path / "v4"
    (output / "run_config.json").write_text("{}")

    missing_threshold = summarize_run(output)

    assert missing_threshold["qualified_count"] is None
    assert missing_threshold["median_validation_abs_rankic"] is None

    (output / "run_config.json").write_text(json.dumps({"final_rankic_threshold": 0.0}))
    rows = list(csv.DictReader((output / "validation_rankic.csv").open()))
    rows[0]["screening_rank_ic"] = "0.0"
    rows[0]["same_direction"] = "1"
    with (output / "validation_rankic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    assert summarize_run(output)["qualified_count"] == 1

    rows[0]["same_direction"] = "0"
    with (output / "validation_rankic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert summarize_run(output)["qualified_count"] == 0

    rows[0].pop("same_direction")
    with (output / "validation_rankic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert summarize_run(output)["qualified_count"] == 0


def test_resource_and_runtime_metrics_require_explicit_units_and_take_max(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    output = tmp_path / "v4"
    (output / "resource_summary.json").write_text(
        json.dumps({"peak_memory_mb": 128.0, "workers": [{"max_rss_bytes": 512 * 1024**2}]})
    )
    with (output / "resources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["peak_rss_mb"])
        writer.writeheader()
        writer.writerow({"peak_rss_mb": "768"})
    with (output / "gp_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["runtime_minutes"])
        writer.writeheader()
        writer.writerow({"runtime_minutes": "2"})

    report = summarize_run(output)

    assert report["peak_memory_mb"] == pytest.approx(768.0)
    assert report["runtime_seconds"] is None


def test_legacy_factor_metrics_fallback_requires_the_v3_selected_factor_schema(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with (output / "factor_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unrelated", "value"])
        writer.writeheader()
        writer.writerow({"unrelated": "row", "value": "1"})

    assert summarize_run(output)["selected_count"] is None

    with (output / "factor_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "expression", "evolution_rank_ic", "screening_rank_ic"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "gp_0001",
                "expression": '["rank","feature_001"]',
                "evolution_rank_ic": "0.04",
                "screening_rank_ic": "0.05",
            }
        )

    assert summarize_run(output)["selected_count"] == 1

    with (output / "factor_metrics.csv").open("w", newline="") as handle:
        csv.DictWriter(
            handle,
            fieldnames=["name", "expression", "screening_rank_ic"],
        ).writeheader()
    assert summarize_run(output)["selected_count"] == 0


def test_runtime_and_memory_ignore_negative_values(tmp_path):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    output = tmp_path / "v4"
    with (output / "gp_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["runtime_seconds"])
        writer.writeheader()
        writer.writerow({"runtime_seconds": "-2"})
    (output / "resource_summary.json").write_text(
        json.dumps({"peak_memory_mb": -1, "workers": [{"max_rss_bytes": -1024}]})
    )

    negative_only = summarize_run(output)

    assert negative_only["runtime_seconds"] is None
    assert negative_only["peak_memory_mb"] is None

    with (output / "gp_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["runtime_seconds"])
        writer.writeheader()
        writer.writerows([{"runtime_seconds": "-2"}, {"runtime_seconds": "7"}])
    (output / "resource_summary.json").write_text(
        json.dumps({"peak_memory_mb": -1, "workers": [{"max_rss_bytes": 256 * 1024**2}]})
    )

    mixed = summarize_run(output)

    assert mixed["runtime_seconds"] == pytest.approx(7.0)
    assert mixed["peak_memory_mb"] == pytest.approx(256.0)

    with (output / "gp_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["runtime_seconds"])
        writer.writeheader()
        writer.writerows([{"runtime_seconds": "1e308"}, {"runtime_seconds": "1e308"}])
    assert summarize_run(output)["runtime_seconds"] is None


def test_cli_output_is_deterministic(tmp_path, capsys):
    write_fake_run(tmp_path / "v3", qualified=1, selected=1)
    write_fake_run(tmp_path / "v4", qualified=1, selected=1)
    argv = ["--v3-output", str(tmp_path / "v3"), "--v4-output", str(tmp_path / "v4")]

    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    second = capsys.readouterr().out

    assert first == second

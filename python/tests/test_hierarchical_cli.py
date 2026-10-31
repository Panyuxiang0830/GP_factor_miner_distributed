import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from conftest import write_eod_pickles
from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.hierarchical_cli import (
    _screening_candidates,
    _screening_progress_logger,
    build_parser,
    main,
)
from gp_factor_miner.hierarchical_screening import screen_hierarchical_factors


def test_parser_supports_output_override_and_pilot_smoke():
    args = build_parser().parse_args(
        ["--data-dir", "train", "--output-dir", "out", "--smoke"]
    )

    assert args.smoke is True
    assert args.output_dir == Path("out")


def test_parser_defaults_to_dedicated_eod_output_directory():
    args = build_parser().parse_args([])

    assert args.output_dir == Path("/hy-tmp/data_pyx/GP_output_eod/run_01")


def test_screening_progress_logger_reports_stage_counts(capsys):
    logger = _screening_progress_logger()
    logger({"event": "validation_progress", "completed": 100, "total": 205})
    logger({"event": "validation_completed", "completed": 205, "total": 205})
    logger(
        {
            "event": "correlation_progress",
            "completed": 100,
            "total": 205,
            "selected": 12,
        }
    )
    logger(
        {
            "event": "correlation_completed",
            "completed": 205,
            "total": 205,
            "selected": 18,
        }
    )

    assert capsys.readouterr().out.splitlines() == [
        "[screening][validation] progress 100/205",
        "[screening][validation] completed 205/205",
        "[screening][correlation] progress 100/205 selected=12",
        "[screening][correlation] completed 205/205 selected=18",
    ]


def test_smoke_writes_depth_complete_v4_outputs(tmp_path):
    output_dir = tmp_path / "data-disk" / "v4-output"
    data_file, label_file = write_eod_pickles(tmp_path, time_ids=range(40))

    assert main(
        [
            "--data-file",
            str(data_file),
            "--label-file",
            str(label_file),
            "--output-dir",
            str(output_dir),
            "--smoke",
        ]
    ) == 0

    expected = {
        "gene_pool_depth_01.json",
        "gene_pool_depth_02.json",
        "gene_pool_depth_03.json",
        "pca_qd_archive.json",
        "validation_rankic.csv",
        "correlation_decisions.csv",
        "selected_gp_factors.json",
        "run_config.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}

    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["checkpoint_version"] == 4
    assert run_config["completed_depths"] == [1, 2, 3]
    assert run_config["pca_similarity_threshold"] == 0.90
    assert run_config["final_rankic_threshold"] == 0.07
    assert run_config["final_correlation_threshold"] == 0.75
    assert run_config["final_factor_count"] is None

    archive = json.loads((output_dir / "pca_qd_archive.json").read_text())
    assert archive
    assert all("signature" not in record for record in archive)

    checkpoint_path = output_dir / "eval" / "run_01" / "completed.pkl"
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert checkpoint["version"] == 4
    assert [stage.depth for stage in checkpoint["result"].stages] == [1, 2, 3]

    checkpoint_mtime = checkpoint_path.stat().st_mtime_ns
    selected_before_resume = (output_dir / "selected_gp_factors.json").read_bytes()
    assert main(
        [
            "--data-file",
            str(data_file),
            "--label-file",
            str(label_file),
            "--output-dir",
            str(output_dir),
            "--smoke",
        ]
    ) == 0
    assert checkpoint_path.stat().st_mtime_ns == checkpoint_mtime
    assert (output_dir / "selected_gp_factors.json").read_bytes() == selected_before_resume


def test_v4_refuses_v3_checkpoint_before_writing_setup_outputs(tmp_path):
    output_dir = tmp_path / "v3-output"
    checkpoint_dir = output_dir / "eval" / "run_01"
    checkpoint_dir.mkdir(parents=True)
    data_file, label_file = write_eod_pickles(tmp_path, time_ids=range(40))
    checkpoint_path = checkpoint_dir / "completed.pkl"
    checkpoint_path.write_bytes(pickle.dumps({"version": 3, "result": object()}))

    with pytest.raises(RuntimeError, match="new output directory"):
        main(
            [
                "--data-file",
                str(data_file),
                "--label-file",
                str(label_file),
                "--output-dir",
                str(output_dir),
                "--smoke",
            ]
        )

    assert not (output_dir / "eval" / "setup").exists()
    assert pickle.loads(checkpoint_path.read_bytes())["version"] == 3


def test_v4_refuses_mixed_v3_marker_and_valid_v4_checkpoint_without_writes(
    tmp_path,
):
    output_dir = tmp_path / "mixed-output"
    data_file, label_file = write_eod_pickles(tmp_path, time_ids=range(40))
    argv = [
        "--data-file",
        str(data_file),
        "--label-file",
        str(label_file),
        "--output-dir",
        str(output_dir),
        "--smoke",
    ]
    assert main(argv) == 0
    marker = output_dir / "selected_raw_features.json"
    marker.write_text("v3 marker", encoding="utf-8")
    protected = [
        marker,
        output_dir / "run_config.json",
        output_dir / "eval" / "setup" / "evolution_profile.csv",
        output_dir / "eval" / "run_01" / "completed.pkl",
    ]
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected
    }

    with pytest.raises(RuntimeError, match="separate from V3"):
        main(argv)

    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected
    } == before


@pytest.mark.parametrize("payload", ({"version": 3}, {"version": 4}))
def test_v4_scans_extra_run_directories_before_writing(tmp_path, payload):
    output_dir = tmp_path / "output"
    legacy_checkpoint = output_dir / "eval" / "run_99" / "completed.pkl"
    legacy_checkpoint.parent.mkdir(parents=True)
    data_file, label_file = write_eod_pickles(tmp_path, time_ids=range(40))
    legacy_checkpoint.write_bytes(pickle.dumps(payload))
    before = (legacy_checkpoint.stat().st_mtime_ns, legacy_checkpoint.read_bytes())

    with pytest.raises(RuntimeError, match="new output directory"):
        main(
            [
                "--data-file",
                str(data_file),
                "--label-file",
                str(label_file),
                "--output-dir",
                str(output_dir),
                "--smoke",
            ]
        )

    assert not (output_dir / "eval" / "setup").exists()
    assert (legacy_checkpoint.stat().st_mtime_ns, legacy_checkpoint.read_bytes()) == before


def test_archive_only_screening_candidate_keeps_original_metrics(tmp_path):
    original = Candidate(
        expression="archive_only",
        rank_ic=-0.071,
        abs_rank_ic=0.071,
        valid_ratio=0.93,
        missing_ratio=0.07,
        depth=0,
        nodes=1,
        fitness=0.042,
        segment_rank_ic_std=0.019,
        same_sign_ratio=0.75,
        segment_rank_ics=(-0.04, -0.06, -0.08, -0.10),
    )
    result = SimpleNamespace(
        stages=(
            SimpleNamespace(
                gene_pool=(
                    replace(
                        original,
                        fitness=0.001,
                        segment_rank_ic_std=0.0,
                        same_sign_ratio=1.0,
                    ),
                )
            ),
        ),
        archive_records=(
            SimpleNamespace(
                key='"archive_only"',
                expression="archive_only",
                abs_rank_ic=0.071,
                nodes=1,
                stage_depth=1,
            ),
        ),
        archive_candidates=(original,),
    )

    candidates = _screening_candidates([result], {}, None)
    assert candidates == (original,)

    target = pd.DataFrame([[1.0, 2.0, 3.0]] * 4)
    selected = screen_hierarchical_factors(
        candidates,
        {"archive_only": target.iloc[:, ::-1].set_axis(target.columns, axis=1)},
        target,
        tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
    )
    validation = pd.read_csv(tmp_path / "validation_rankic.csv")
    assert validation.iloc[0]["evolution_rank_ic"] == pytest.approx(-0.071)
    assert validation.iloc[0]["fitness"] == pytest.approx(0.042)
    assert selected[0].fitness == pytest.approx(0.042)
    assert selected[0].segment_rank_ic_std == pytest.approx(0.019)
    assert selected[0].same_sign_ratio == pytest.approx(0.75)

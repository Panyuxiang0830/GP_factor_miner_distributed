from pathlib import Path
import json

from conftest import write_panel
import gp_factor_miner.cli
from gp_factor_miner.cli import _progress_bar, main


def test_progress_bar_shows_chunk_percentage():
    assert _progress_bar(1_000, 10_000, width=10) == "[#---------]  10.0%"


def test_v3_parser_and_entrypoint_remain_available():
    args = gp_factor_miner.cli.build_parser().parse_args(
        ["--data-dir", "train", "--output-dir", "out", "--smoke"]
    )

    assert args.smoke is True


def test_smoke_cli_writes_complete_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    write_panel(data_dir, time_ids=range(1000))

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--smoke",
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "selected_raw_features.json",
        "selected_gp_factors.json",
        "factor_metrics.csv",
        "gp_history.csv",
        "run_config.json",
        "eval",
    }
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["processed_evolution_time_count"] == 256
    assert run_config["processed_screening_time_count"] == 64
    assert (output_dir / "eval" / "setup" / "evolution_profile.csv").exists()
    assert (output_dir / "eval" / "run_01" / "generation_01").exists()

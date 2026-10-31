import pyarrow.parquet as pq

from conftest import write_eod_pickles, write_panel
from gp_factor_miner.data import (
    compute_eod_time_split_from_ids,
    compute_time_split,
    discover_schema,
    load_eod_panel_dataset,
    load_research_rows,
    split_eod_panels,
)


def test_eod_split_uses_fixed_evolution_and_screening_years():
    split = compute_eod_time_split_from_ids(
        ["20151231", "20160104", "20211231", "20220104", "20221230", "20230103"]
    )

    assert split.evolution_time_ids == ("20160104", "20211231")
    assert split.screening_time_ids == ("20220104", "20221230")
    assert split.n_time_ids == 6


def test_split_uses_first_40_and_next_10_percent(tmp_path):
    write_panel(tmp_path, time_ids=range(10))

    split = compute_time_split(tmp_path)

    assert split.evolution_time_ids == (0, 1, 2, 3)
    assert split.screening_time_ids == (4,)
    assert split.evolution_end == 3
    assert split.screening_end == 4


def test_loader_never_returns_sealed_half(tmp_path):
    write_panel(tmp_path, time_ids=range(10))
    split = compute_time_split(tmp_path)

    frame = load_research_rows(
        tmp_path,
        split,
        ["time_id", "asset_id", "target"],
    )

    assert frame["time_id"].max() == 4
    assert set(frame["time_id"].unique()) == {0, 1, 2, 3, 4}


def test_loader_projects_only_requested_columns(tmp_path):
    parquet_path = write_panel(tmp_path, time_ids=range(10))
    split = compute_time_split(tmp_path)

    frame = load_research_rows(tmp_path, split, ["time_id", "target"])

    assert list(frame.columns) == ["time_id", "target"]
    assert pq.read_schema(parquet_path).names != list(frame.columns)


def test_discover_schema_finds_sorted_features(tmp_path):
    write_panel(tmp_path)

    schema = discover_schema(tmp_path)

    assert schema.feature_names == ("feature_000", "feature_001")
    assert schema.required_names == ("time_id", "asset_id", "target")


def test_eod_loader_aligns_suffixes_masks_universe_and_uses_fixed_windows(tmp_path):
    data_file, label_file = write_eod_pickles(tmp_path, time_ids=range(40))
    dataset = load_eod_panel_dataset(data_file, label_file)

    assert dataset.feature_names[:2] == ("close", "amount")
    assert len(dataset.feature_names) == 14
    assert dataset.common_date_count == 40
    assert dataset.common_asset_count == 8
    assert len(dataset.target.index) == 20
    assert list(dataset.target.columns[:3]) == ["000001", "000002", "000003"]

    evolution, evolution_target, screening, screening_target = split_eod_panels(
        dataset
    )
    assert len(evolution_target) == 16
    assert len(screening_target) == 4
    assert evolution["close"].shape == (16, 8)
    assert screening["amount"].shape == (4, 8)

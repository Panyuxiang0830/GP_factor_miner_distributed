"""Shared fixtures: synthetic panels written in the export format.

Every parity test compares libgpp_native with the unchanged course Python
implementation (python/src/gp_factor_miner) on the same float32 values.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "real_product" / "python"))

from gpp.export import write_panels  # noqa: E402

from gpp.synthetic import FIELDS, make_frames  # noqa: E402,F401


@pytest.fixture(scope="session")
def native_lib():
    from gpp import native

    try:
        native.lib()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return native


@pytest.fixture(scope="session")
def evolution_panels(tmp_path_factory):
    frames, target = make_frames(150, 64, seed=11)
    out = tmp_path_factory.mktemp("evolution")
    write_panels(out, frames, target, period="evolution")
    return out, frames, target


@pytest.fixture(scope="session")
def screening_panels(tmp_path_factory):
    frames, target = make_frames(120, 64, seed=12, start="2022-01-04")
    out = tmp_path_factory.mktemp("screening")
    write_panels(out, frames, target, period="screening")
    return out, frames, target

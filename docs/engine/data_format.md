# Panel data format

`python/scripts/export_panels.py` writes one directory per data set. Data never enter the repository.

```
<dir>/meta.json             metadata (below)
<dir>/fields/<field>.f32    one raw field
<dir>/target.f32            forward-return label, masked to tradable, non-ST rows
```

## Binary layout

- Each `.f32` file is a raw **little-endian float32** array of `T × N` values. There is no header.
- The layout is **date-major**: day `t`, asset `s` is stored at element `t * N + s`.
- A missing value is IEEE NaN.

## meta.json

| key | meaning |
|---|---|
| `format_version` | currently 1 |
| `period` | `evolution` (2016–2021) or `screening` (2022) |
| `T`, `N` | number of dates and assets |
| `dates`, `assets` | axis labels, in file order |
| `fields` | field names; each has a matching `fields/<name>.f32` |
| `dtype`, `layout` | `float32`, `date_major` |

## Cleaning

Cleaning reuses `gp_factor_miner.data.load_eod_panel_dataset`:

- features are aligned to the dates and assets shared with the label, then cast to float32;
- features are not masked;
- the target is masked by `trade_status` and `ST`.

## Current exports (on gpushare, `/hy-tmp/data_pyx/GP_distributed/`)

| directory | T | N | fields | size |
|---|---|---|---|---|
| `sample_120x300` | 120 | 300 | 132 | about 20 MB |
| `evolution_full` | 1461 | 5876 | 132 | 4.3 GB |

## Reference outputs

`python/scripts/export_reference.py` writes the outputs of the original pandas implementation:

- `normalize` mode: normalization cases;
- `values` mode: factor panels as float64, daily RankIC and candidate statistics.

Fixtures built from mined factors stay in the data directory. Only the crafted cases (`tests/data/normalize_cases.json`) are committed.

#!/usr/bin/env python3
"""Build a private manifest without copying the large shared R files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
from gp_factor_miner.expressions import expression_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parent / "runs"
    out = args.out.resolve()
    if not out.is_relative_to(runs.resolve()) or out.exists():
        parser.error("--out must be a new directory inside pca/runs")
    export = args.export_dir.resolve()
    meta = json.loads((export / "R/meta.json").read_text())
    records = []
    for line_no, line in enumerate((export / "scores.jsonl").read_text().splitlines()):
        entry = json.loads(line)
        if not entry.get("valid", True) or entry.get("constant_R", False):
            continue
        quality = entry.get("abs_rank_ic")
        records.append(dict(
            index=entry["index"], score_line=line_no,
            key=expression_key(entry["expression"]),
            quality=quality if quality is not None else 0.0,
            candidate=bool(entry["candidate"]) and quality is not None,
            path=str(export / "R" / f"chunk_{entry['chunk']:05d}.f32"),
            slot=entry["slot"],
        ))
    seen = set()
    for entry in records:
        location = entry["path"], entry["slot"]
        if location in seen:
            raise ValueError("duplicate R location")
        seen.add(location)
    if args.init.stat().st_size != meta["N"] * 8:
        raise ValueError("incorrect initial vector size")
    if (export / "signatures.f32").stat().st_size != len(records) * meta["rows"] * 4:
        raise ValueError("incorrect reference signature size")
    manifest = dict(rows=meta["rows"], cols=meta["N"], initial=str(args.init.resolve()),
                    provided_signatures=str(export / "signatures.f32"),
                    records=records, source_meta=meta)
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(records)} R records; large inputs remain in their original locations.")


if __name__ == "__main__":
    main()

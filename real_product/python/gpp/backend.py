"""In-process batch backend for the hierarchical engine (gpp.hier_engine).

Implements the protocol of pca/gp_bridge.py's CppGPBackend
(``identity``, ``validate_inputs``, ``evaluate``, ``consider``) without
subprocesses or files: one NativeEngine holds the evolution panels for the
whole run, and one NativeArchive mirrors the Python PCAQDArchive.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gp_factor_miner.expressions import expression_depth, expression_key, expression_nodes, normalize_expression
from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.pca_qd import QDDecision

from .native import NativeArchive, NativeEngine, library_path


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeBackend:
    def __init__(self, panel_dir, config, *, dates=None, stocks=None):
        self.panel_dir = Path(panel_dir).resolve()
        self.meta = json.loads((self.panel_dir / "meta.json").read_text())
        self.T = int(dates or self.meta["T"])
        self.N = int(stocks or self.meta["N"])
        if not 2 <= self.T <= self.meta["T"] or not 2 <= self.N <= self.meta["N"]:
            raise ValueError("invalid backend shape")
        self.config = config
        self.identity = dict(
            kind="gpp-native-v1",
            panel_dir=str(self.panel_dir),
            meta_sha256=_sha256(self.panel_dir / "meta.json"),
            dates=self.T,
            stocks=self.N,
            library_sha256=_sha256(library_path()),
            signature_step=config.pca_sample_step,
            power_steps=config.pca_power_iterations,
            gpus=config.gpus,
        )
        self._engine = None
        self._fitness = None
        self._seed = None
        self._archive = None
        self._archive_keys = None
        self.calls = []

    # -- protocol -------------------------------------------------------
    def validate_inputs(self, data_dict, target, config):
        if config.pca_sample_step < 1 or (self.T + config.pca_sample_step - 1) // config.pca_sample_step < 2:
            raise ValueError("not enough sampled dates")
        # The production CLI passes no frames (the GP loop never reads them
        # with a backend); tests pass DataFrames and get an exact check.
        if target is not None and hasattr(target, "to_numpy"):
            frames = self.frames(list(data_dict))
            for key, frame in [*data_dict.items(), ("__target__", target)]:
                reference = frames[1] if key == "__target__" else frames[0][key]
                if not np.array_equal(frame.to_numpy(np.float32), reference, equal_nan=True):
                    raise ValueError("Python frames and exported panels differ: " + key)

    def frames(self, fields):
        def load(path):
            values = np.memmap(path, mode="r", dtype="<f4", shape=(self.meta["T"], self.meta["N"]))
            return np.asarray(values[: self.T, : self.N])

        return {name: load(self.panel_dir / "fields" / f"{name}.f32") for name in fields}, load(
            self.panel_dir / "target.f32"
        )

    def _ensure_engine(self, evaluation_config, pca_seed):
        fitness = asdict(evaluation_config)
        if self._engine is None or fitness != self._fitness:
            c = self.config
            if self._engine is not None:
                self._engine.close()
            self._engine = NativeEngine(
                self.panel_dir, dates=self.T, stocks=self.N, threads=c.engine_threads,
                memory_budget_gb=c.memory_budget_gb, max_chunk=c.engine_max_chunk,
                signature_step=c.pca_sample_step, power_steps=c.pca_power_iterations, fitness=fitness,
                gpus=c.gpus, streams_per_gpu=c.streams_per_gpu,
            )
            self._fitness = fitness
            self._seed = None
        if self._seed != pca_seed:
            # pca/gp_bridge.py _initial: default_rng(seed).normal(size=N); the
            # solver normalizes it like factor_signature does.
            self._engine.set_initial(np.random.default_rng(pca_seed).normal(size=self.N))
            self._seed = pca_seed
        return self._engine

    def evaluate(self, expressions, evaluation_config, config, pca_seed):
        if not expressions:
            return []
        engine = self._ensure_engine(evaluation_config, pca_seed)
        normalized = [normalize_expression(x) for x in expressions]
        stats, signatures, flags = engine.evaluate(normalized)
        segments = engine.segments
        result = []
        for i, expression in enumerate(normalized):
            if flags[i] & 1 or not flags[i] & 2:
                result.append((None, None))
                continue
            row = stats[i]
            for value in row[:6]:
                if not math.isfinite(value):
                    raise RuntimeError("nonfinite native candidate statistic")
            candidate = Candidate(
                expression=expression,
                rank_ic=float(row[0]),
                abs_rank_ic=abs(float(row[0])),
                valid_ratio=float(row[1]),
                missing_ratio=float(row[2]),
                depth=expression_depth(expression),
                nodes=expression_nodes(expression),
                fitness=float(row[3]),
                segment_rank_ic_std=float(row[4]),
                same_sign_ratio=float(row[5]),
                segment_rank_ics=tuple(float(x) for x in row[8 : 8 + segments]),
            )
            signature = signatures[i].copy() if flags[i] & 4 else None
            result.append((candidate, signature))
        self.calls.append(dict(kind="evaluate", count=len(normalized), **engine.stats()))
        return result

    def _sync_archive(self, archive, dim):
        keys = list(archive._records)
        if self._archive is not None and self._archive.dim == dim and self._archive_keys == set(keys):
            return
        if self._archive is not None:
            self._archive.close()
        self._archive = NativeArchive(dim, archive.similarity_threshold, archive.replacement_margin,
                                      gpus=self.config.gpus)
        records = [archive._records[key] for key in keys]
        signatures = np.asarray([r.signature for r in records], dtype=np.float32).reshape(len(records), dim)
        self._archive.reset(keys, [r.abs_rank_ic for r in records], signatures)
        self._archive_keys = set(keys)

    def consider(self, archive, records, config, pca_seed):
        if not records:
            return []
        dim = int(np.asarray(records[0].signature).size)
        self._sync_archive(archive, dim)
        signatures = np.asarray([r.signature for r in records], dtype=np.float32).reshape(len(records), dim)
        decisions = self._archive.consider(
            [r.key for r in records], [r.abs_rank_ic for r in records], signatures, block=config.pca_qd_block_size
        )
        current = archive._records
        answer = []
        for record, (action, similarity, closest, replaced) in zip(records, decisions):
            if action == "replace":
                if replaced == record.key:
                    current[record.key] = record          # PCAQDArchive keeps the position
                else:
                    del current[replaced]
                    current[record.key] = record
                self._archive_keys.discard(replaced)
                self._archive_keys.add(record.key)
            elif action == "admit":
                current[record.key] = record
                self._archive_keys.add(record.key)
            answer.append(QDDecision(action, similarity, closest, replaced))
        if len(current) != self._archive.size():
            raise RuntimeError("native/Python archive size mismatch")
        self.calls.append(dict(kind="consider", count=len(records), **self._archive.stats()))
        return answer

    def close(self):
        if self._engine is not None:
            self._engine.close()
        if self._archive is not None:
            self._archive.close()


def expression_keys(expressions):
    return [expression_key(x) for x in expressions]

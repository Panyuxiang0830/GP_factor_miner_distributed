"""ctypes binding of libgpp_native (real_product/src/c_api.cpp).

The library is loaded once per process; panels, target ranks and the archive
matrix stay in native memory for the whole run, so no batch goes through
files (the course bridge wrote a request JSON, launched mpirun, and read
candidates.json + signatures.f32 back for every chunk).
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_LIB = None

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_long_p = ctypes.POINTER(ctypes.c_long)
_c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)


def library_path() -> Path:
    explicit = os.environ.get("GPP_NATIVE_LIB")
    if explicit:
        return Path(explicit)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for build in ("build", "build-cuda", "build-cpu"):
        candidate = _ROOT / build / f"libgpp_native{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "libgpp_native not found; build real_product (scripts/build.sh) or set GPP_NATIVE_LIB"
    )


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    L = ctypes.CDLL(str(library_path()))
    err = ctypes.c_char_p
    L.gpp_engine_open.restype = ctypes.c_void_p
    L.gpp_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, err, ctypes.c_int]
    L.gpp_engine_signature_rows.argtypes = [ctypes.c_void_p]
    L.gpp_engine_set_initial.argtypes = [ctypes.c_void_p, _c_double_p, ctypes.c_int, err, ctypes.c_int]
    L.gpp_engine_evaluate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, _c_double_p,
                                      _c_float_p, _c_int_p, err, ctypes.c_int]
    L.gpp_engine_plan.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, err, ctypes.c_int]
    L.gpp_engine_stats.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    L.gpp_engine_close.argtypes = [ctypes.c_void_p]
    L.gpp_archive_new.restype = ctypes.c_void_p
    L.gpp_archive_new.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_char_p, err, ctypes.c_int]
    L.gpp_archive_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, _c_double_p, _c_float_p, err,
                                    ctypes.c_int]
    L.gpp_archive_consider.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, _c_double_p, _c_float_p,
                                       ctypes.c_int, _c_int_p, _c_double_p, _c_int_p, _c_int_p, _c_int_p, err,
                                       ctypes.c_int]
    L.gpp_archive_size.argtypes = [ctypes.c_void_p]
    L.gpp_archive_stats.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    L.gpp_archive_close.argtypes = [ctypes.c_void_p]
    L.gpp_screen.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                             ctypes.c_int, _c_double_p, _c_int_p, ctypes.c_char_p, _c_double_p, _c_double_p,
                             _c_double_p, _c_int_p, _c_int_p, ctypes.c_char_p, ctypes.c_int, _c_int_p, _c_int_p,
                             _c_ubyte_p, _c_double_p, _c_int_p, _c_long_p, ctypes.c_char_p, ctypes.c_int, err,
                             ctypes.c_int]
    _LIB = L
    return L


def has_cuda() -> bool:
    return bool(lib().gpp_has_cuda())


def _ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _check(code, buffer):
    if code:
        raise RuntimeError(buffer.value.decode("utf-8", "replace"))


def _keys_blob(keys) -> bytes:
    return b"".join(key.encode("utf-8") + b"\0" for key in keys)


def _expressions_json(expressions) -> bytes:
    return json.dumps(list(expressions), separators=(",", ":"), allow_nan=False).encode("utf-8")


class NativeEngine:
    """Shared-memory batch evaluator on one exported panel directory."""

    def __init__(self, panel_dir, *, dates=0, stocks=0, threads=0, memory_budget_gb=0.0, max_chunk=2000,
                 signature_step=5, power_steps=4, fitness=None, gpus=0, streams_per_gpu=4, score_begin=0):
        self.options = dict(threads=threads, memory_budget_gb=memory_budget_gb, max_chunk=max_chunk,
                            signature_step=signature_step, power_steps=power_steps, fitness=dict(fitness or {}),
                            gpus=gpus, streams_per_gpu=streams_per_gpu, score_begin=score_begin)
        error = ctypes.create_string_buffer(4096)
        self._h = lib().gpp_engine_open(str(panel_dir).encode(), int(dates), int(stocks),
                                        json.dumps(self.options).encode(), error, len(error))
        if not self._h:
            raise RuntimeError(error.value.decode())
        self.signature_rows = lib().gpp_engine_signature_rows(self._h)
        self.segments = int(self.options["fitness"].get("stability_segments", 4))

    def set_initial(self, initial):
        values = np.ascontiguousarray(initial, dtype=np.float64)
        error = ctypes.create_string_buffer(4096)
        _check(lib().gpp_engine_set_initial(self._h, _ptr(values, ctypes.c_double), len(values), error, len(error)),
               error)

    def evaluate(self, expressions):
        """Returns (stats [n, 8+K], signatures [n, m] float32, flags [n])."""
        n = len(expressions)
        width = 8 + self.segments
        stats = np.empty((n, width), dtype=np.float64)
        signatures = np.zeros((n, max(self.signature_rows, 1)), dtype=np.float32)
        flags = np.zeros(n, dtype=np.int32)
        if n == 0:
            return stats, signatures[:, : self.signature_rows], flags
        error = ctypes.create_string_buffer(4096)
        code = lib().gpp_engine_evaluate(
            self._h, _expressions_json(expressions), n, self.segments, _ptr(stats, ctypes.c_double),
            _ptr(signatures, ctypes.c_float) if self.signature_rows else None, _ptr(flags, ctypes.c_int), error,
            len(error),
        )
        _check(code, error)
        return stats, signatures[:, : self.signature_rows], flags

    def plan(self, expressions):
        out = ctypes.create_string_buffer(1 << 20)
        error = ctypes.create_string_buffer(4096)
        _check(lib().gpp_engine_plan(self._h, _expressions_json(expressions), out, len(out), error, len(error)), error)
        return json.loads(out.value.decode())

    def stats(self):
        out = ctypes.create_string_buffer(8192)
        lib().gpp_engine_stats(self._h, out, len(out))
        return json.loads(out.value.decode())

    def close(self):
        if getattr(self, "_h", None):
            lib().gpp_engine_close(self._h)
            self._h = None

    def __del__(self):
        self.close()


ACTIONS = ("admit", "replace", "reject")


class NativeArchive:
    """QD archive kept in native memory; slots map to keys on the Python side."""

    def __init__(self, dim, threshold, margin, *, top_k=4, gpus=0):
        error = ctypes.create_string_buffer(4096)
        options = json.dumps(dict(top_k=top_k, gpus=gpus)).encode()
        self._h = lib().gpp_archive_new(int(dim), float(threshold), float(margin), options, error, len(error))
        if not self._h:
            raise RuntimeError(error.value.decode())
        self.dim = int(dim)
        self.slot_keys: list[str] = []

    def reset(self, keys, quality, signatures):
        keys = list(keys)
        quality = np.ascontiguousarray(quality, dtype=np.float64)
        signatures = np.ascontiguousarray(signatures, dtype=np.float32).reshape(len(keys), self.dim)
        error = ctypes.create_string_buffer(4096)
        _check(lib().gpp_archive_reset(self._h, len(keys), _keys_blob(keys), _ptr(quality, ctypes.c_double),
                                       _ptr(signatures, ctypes.c_float), error, len(error)), error)
        self.slot_keys = keys

    def consider(self, keys, quality, signatures, block=4096):
        keys = list(keys)
        n = len(keys)
        quality = np.ascontiguousarray(quality, dtype=np.float64)
        signatures = np.ascontiguousarray(signatures, dtype=np.float32).reshape(n, self.dim)
        action = np.empty(n, dtype=np.int32)
        similarity = np.empty(n, dtype=np.float64)
        closest = np.empty(n, dtype=np.int32)
        replaced = np.empty(n, dtype=np.int32)
        new_slot = np.empty(n, dtype=np.int32)
        if n == 0:
            return []
        error = ctypes.create_string_buffer(4096)
        _check(lib().gpp_archive_consider(
            self._h, n, _keys_blob(keys), _ptr(quality, ctypes.c_double), _ptr(signatures, ctypes.c_float),
            int(block), _ptr(action, ctypes.c_int), _ptr(similarity, ctypes.c_double), _ptr(closest, ctypes.c_int),
            _ptr(replaced, ctypes.c_int), _ptr(new_slot, ctypes.c_int), error, len(error)), error)
        out = []
        for i in range(n):
            if new_slot[i] >= 0:
                if new_slot[i] != len(self.slot_keys):
                    raise RuntimeError("native archive slot numbering out of sync")
                self.slot_keys.append(keys[i])
            out.append((
                ACTIONS[action[i]],
                float(similarity[i]),
                self.slot_keys[closest[i]] if closest[i] >= 0 else None,
                self.slot_keys[replaced[i]] if replaced[i] >= 0 else None,
            ))
        return out

    def size(self):
        return lib().gpp_archive_size(self._h)

    def stats(self):
        out = ctypes.create_string_buffer(4096)
        lib().gpp_archive_stats(self._h, out, len(out))
        return json.loads(out.value.decode())

    def close(self):
        if getattr(self, "_h", None):
            lib().gpp_archive_close(self._h)
            self._h = None

    def __del__(self):
        self.close()


def native_screen(panel_dir, expressions, evolution_rank_ic, nodes, keys, *, dates=0, stocks=0, options=None):
    """One call of the C++ screening; returns a dict of arrays."""
    n = len(expressions)
    evolution_rank_ic = np.ascontiguousarray(evolution_rank_ic, dtype=np.float64)
    nodes = np.ascontiguousarray(nodes, dtype=np.int32)
    mean = np.empty(n)
    valid_ratio = np.empty(n)
    missing_ratio = np.empty(n)
    valid_times = np.empty(n, dtype=np.int32)
    status = np.empty(n, dtype=np.int32)
    error_len = 256
    errors = ctypes.create_string_buffer(max(1, n * error_len))
    order = np.empty(max(n, 1), dtype=np.int32)
    n_order = ctypes.c_int(0)
    selected = np.empty(max(n, 1), dtype=np.uint8)
    max_corr = np.empty(max(n, 1))
    corr_with = np.empty(max(n, 1), dtype=np.int32)
    scanned = ctypes.c_long(0)
    stats = ctypes.create_string_buffer(8192)
    error = ctypes.create_string_buffer(4096)
    _check(lib().gpp_screen(
        str(panel_dir).encode(), int(dates), int(stocks), json.dumps(options or {}).encode(),
        _expressions_json(expressions), n, _ptr(evolution_rank_ic, ctypes.c_double), _ptr(nodes, ctypes.c_int),
        _keys_blob(keys), _ptr(mean, ctypes.c_double), _ptr(valid_ratio, ctypes.c_double),
        _ptr(missing_ratio, ctypes.c_double), _ptr(valid_times, ctypes.c_int), _ptr(status, ctypes.c_int), errors,
        error_len, _ptr(order, ctypes.c_int), ctypes.byref(n_order), _ptr(selected, ctypes.c_ubyte),
        _ptr(max_corr, ctypes.c_double), _ptr(corr_with, ctypes.c_int), ctypes.byref(scanned), stats, len(stats),
        error, len(error)), error)
    k, s = n_order.value, scanned.value
    raw = errors.raw
    messages = [raw[i * error_len:(i + 1) * error_len].split(b"\0", 1)[0].decode("utf-8", "replace") for i in range(n)]
    return dict(mean=mean, valid_ratio=valid_ratio, missing_ratio=missing_ratio, valid_times=valid_times,
                status=status, errors=messages, order=order[:k].copy(), selected=selected[:s].astype(bool),
                max_correlation=max_corr[:s].copy(), correlated_with=corr_with[:s].copy(),
                stats=json.loads(stats.value.decode()))

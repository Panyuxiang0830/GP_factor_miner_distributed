from __future__ import annotations

from collections.abc import Iterable

from .expressions import evaluate_expression, normalize_expression


class FactorGenerator:
    def __init__(self, data_dict, max_workers: int = 16):
        self.data_dict = data_dict
        self.max_workers = max_workers
        self.factor_definitions = {}

    def register_factor(self, factor_name: str, expression, description: str = "") -> None:
        if factor_name in self.factor_definitions:
            raise ValueError(f"factor already registered: {factor_name}")
        self.factor_definitions[factor_name] = normalize_expression(expression)

    def compute_factor(self, factor_name: str):
        if factor_name not in self.factor_definitions:
            raise ValueError(f"factor not registered: {factor_name}")
        try:
            return evaluate_expression(self.factor_definitions[factor_name], self.data_dict)
        except Exception as exc:
            raise RuntimeError(f"failed to compute factor {factor_name}") from exc

    def batch_compute(
        self,
        factor_names: Iterable[str],
        max_workers: int | None = None,
        chunksize: int = 1,
    ) -> dict:
        del chunksize
        names = list(factor_names)
        workers = self.max_workers if max_workers is None else min(max_workers, self.max_workers)
        if workers < 1:
            raise ValueError("max_workers must be positive")
        return {name: self.compute_factor(name) for name in names}

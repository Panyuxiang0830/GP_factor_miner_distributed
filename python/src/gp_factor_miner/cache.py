from __future__ import annotations


class ExpressionCache:
    def __init__(self):
        self._values = {}
        self.hits = 0

    def get(self, key):
        if key in self._values:
            self.hits += 1
            return self._values[key]
        return None

    def put(self, key, value) -> None:
        self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)

"""Python numerical reference for validation only, not the production solver."""
import numpy as np


def power_signature(matrix, initial, iterations):
    """FP64 iteration on exactly the exported FP32 R; FP32 output."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.any(matrix):
        raise ValueError("R must be finite and nonzero")

    def unit(vector):
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("zero or nonfinite power-iteration norm")
        return vector / norm

    right = unit(initial.copy())
    for _ in range(iterations):
        left = unit(matrix @ right)
        right = unit(matrix.T @ left)
    signature = unit(matrix @ right)
    if signature[np.argmax(np.abs(signature))] < 0:
        signature *= -1
    return signature.astype(np.float32)

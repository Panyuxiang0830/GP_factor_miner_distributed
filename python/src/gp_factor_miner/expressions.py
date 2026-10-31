from __future__ import annotations

import json
from typing import Any

from .operators import OPERATORS


COMMUTATIVE = {"+", "*"}


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def normalize_expression(expression):
    expression = _as_tuple(expression)
    if not isinstance(expression, tuple):
        return expression
    if not expression:
        raise ValueError("expression tuple cannot be empty")
    operator = expression[0]
    operands = tuple(normalize_expression(item) for item in expression[1:])
    if operator in COMMUTATIVE and len(operands) == 2:
        operands = tuple(sorted(operands, key=repr))
    if operator == "rank" and isinstance(operands[0], tuple) and operands[0][0] == "rank":
        return operands[0]
    if operator in {"-", "/"} and len(operands) == 2 and operands[0] == operands[1]:
        raise ValueError(f"degenerate expression: {operator}(x, x)")
    return (operator, *operands)


def expression_depth(expression) -> int:
    expression = _as_tuple(expression)
    if not isinstance(expression, tuple):
        return 0
    child_depths = [expression_depth(item) for item in expression[1:] if not isinstance(item, (int, float))]
    return 1 + (max(child_depths) if child_depths else 0)


def expression_nodes(expression) -> int:
    expression = _as_tuple(expression)
    if isinstance(expression, (int, float)):
        return 0
    if not isinstance(expression, tuple):
        return 1
    return 1 + sum(expression_nodes(item) for item in expression[1:])


def evaluate_expression(expression, data):
    expression = _as_tuple(expression)
    if isinstance(expression, (int, float)):
        return expression
    if isinstance(expression, str):
        if expression not in data:
            raise ValueError(f"unknown terminal: {expression}")
        return data[expression]
    operator = expression[0]
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator}")
    operands = [evaluate_expression(item, data) for item in expression[1:]]
    return OPERATORS[operator](*operands)


def expression_key(expression) -> str:
    return json.dumps(normalize_expression(expression), separators=(",", ":"))

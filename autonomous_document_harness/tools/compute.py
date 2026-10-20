"""Deterministic computation: sandboxed arithmetic + restricted analysis scripts.

Prefer this over model calls whenever math suffices (mission rule §34.7).
"""

from __future__ import annotations

import ast
import math
import statistics
from typing import Any, Dict

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)
_ALLOWED_FUNCS = {
    "sum",
    "min",
    "max",
    "len",
    "round",
    "sorted",
    "abs",
    "range",
    "enumerate",
    "zip",
    "any",
    "all",
}
_ALLOWED_MODULE_ATTRS = {"math": math, "statistics": statistics}
_RANGE_HARD_CAP = 1_000_000


def safe_eval(expression: str) -> Any:
    """Evaluate a pure arithmetic expression. No names, no calls, no IO."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(
            node,
            (
                ast.Load,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.UAdd,
                ast.USub,
            ),
        ):
            continue
        raise ValueError(f"expression uses forbidden construct: {type(node).__name__}")
    return eval(compile(tree, "<calculate>", "eval"), {"__builtins__": {}})  # noqa: S307


def run_analysis(code: str, data: Any) -> Dict[str, Any]:
    """Run a restricted analysis script with `data` in scope; reads `result`.

    Forbidden: imports, I/O, loops over unbounded ranges, attribute access
    outside math/statistics, dunder access, nested definitions.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid script: {exc}")
    _validate_script(tree)
    scope: Dict[str, Any] = {
        "data": data,
        "math": math,
        "statistics": statistics,
        "result": None,
    }
    import builtins

    for name in _ALLOWED_FUNCS:
        scope[name] = getattr(builtins, name)
    exec(compile(tree, "<analysis>", "exec"), {"__builtins__": {}}, scope)  # noqa: S102
    return {"result": scope.get("result")}


def _validate_script(tree: ast.Module) -> None:
    allowed = (
        (
            ast.Module,
            ast.Expr,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.If,
            ast.For,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.comprehension,
            ast.Subscript,
            ast.Slice,
            ast.Attribute,
            ast.IfExp,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.keyword,
        )
        + _ALLOWED_BINOPS
        + _ALLOWED_UNARY
        + (
            ast.And,
            ast.Or,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Not,
            ast.USub,
            ast.UAdd,
        )
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"script uses forbidden construct: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("dunder attribute access is forbidden")
        if isinstance(node, ast.Call):
            _validate_call(node)


def _validate_call(node: ast.Call) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        if func.id not in _ALLOWED_FUNCS:
            raise ValueError(f"call to '{func.id}' is not allowed")
        if func.id == "range":
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, int)
                    and abs(arg.value) > _RANGE_HARD_CAP
                ):
                    raise ValueError("range() argument exceeds hard cap")
        return
    if isinstance(func, ast.Attribute):
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in _ALLOWED_MODULE_ATTRS:
            return
    raise ValueError("only math/statistics functions and safe builtins are callable")

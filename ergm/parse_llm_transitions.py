"""
Parse Gemma / LLM assistant text into ERGM transition dicts (s, a, s_next).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

# JSON numbers only; LLMs often emit invalid "0.1 + 0.01" inside arrays.
_NUM = r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][+-]?\d+)?"
_BIN_OP = re.compile(rf"({_NUM})\s*([+\-*/])\s*({_NUM})")


def _fold_binary_float_exprs(s: str) -> str:
    """Turn invalid JSON like [0.1 + 0.01, 2 * 3] into numeric literals, left-to-right."""

    def _fmt(v: float) -> str:
        if math.isnan(v) or math.isinf(v):
            return "0.0"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        t = format(v, ".12g")
        if "e" in t.lower() or "E" in t:
            return t
        return t.rstrip("0").rstrip(".") or "0"

    def _apply(m: re.Match[str]) -> str:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        if op == "+":
            v = a + b
        elif op == "-":
            v = a - b
        elif op == "*":
            v = a * b
        else:
            v = a / b if b != 0 else 0.0
        return _fmt(v)

    prev = None
    while prev != s:
        prev = s
        m = _BIN_OP.search(s)
        if not m:
            break
        s = s[: m.start()] + _apply(m) + s[m.end() :]
    return s


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _extract_json_array(text: str) -> str:
    text = _strip_markdown_fences(text)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _as_float_vec(vec: Any, raw_dim: int, item_i: int, key: str) -> list[float]:
    if not isinstance(vec, list):
        raise ValueError(f"Item {item_i}[{key}] must be a list of numbers")
    if len(vec) != raw_dim:
        raise ValueError(f"Item {item_i}[{key}] must have length {raw_dim}, got {len(vec)}")
    out: list[float] = []
    for j, x in enumerate(vec):
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append(float(x))
        elif isinstance(x, str):
            try:
                out.append(float(x))
            except ValueError as e:
                raise ValueError(f"Item {item_i}[{key}][{j}] must be a number") from e
        else:
            raise ValueError(f"Item {item_i}[{key}][{j}] must be a number")
    return out


def parse_transition_array(content: str, raw_dim: int = 8) -> list[dict[str, list[float]]]:
    """
    Parse model output into a list of {"s", "a", "s_next"} each length raw_dim.
    Raises ValueError if structure is invalid.
    """
    blob = _extract_json_array(content)
    blob = _fold_binary_float_exprs(blob)
    data = json.loads(blob)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of transitions")

    out: list[dict[str, list[float]]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} must be an object")
        for key in ("s", "a", "s_next"):
            if key not in row:
                raise ValueError(f"Item {i} missing key {key!r}")
        s_list = _as_float_vec(row["s"], raw_dim, i, "s")
        a_list = _as_float_vec(row["a"], raw_dim, i, "a")
        sn_list = _as_float_vec(row["s_next"], raw_dim, i, "s_next")
        out.append({"s": s_list, "a": a_list, "s_next": sn_list})
    return out


def transitions_to_numpy(
    rows: list[dict[str, list[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (s, a, s_next) each shape (N, D)."""
    if not rows:
        raise ValueError("empty transition list")
    d = len(rows[0]["s"])
    n = len(rows)
    s = np.zeros((n, d), dtype=np.float32)
    a = np.zeros((n, d), dtype=np.float32)
    sn = np.zeros((n, d), dtype=np.float32)
    for i, r in enumerate(rows):
        s[i] = r["s"]
        a[i] = r["a"]
        sn[i] = r["s_next"]
    return s, a, sn

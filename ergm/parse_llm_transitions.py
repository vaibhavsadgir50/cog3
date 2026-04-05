"""
Parse Gemma / LLM assistant text into ERGM transition dicts (s, a, s_next).
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


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
        else:
            raise ValueError(f"Item {item_i}[{key}][{j}] must be a number")
    return out


def parse_transition_array(content: str, raw_dim: int = 8) -> list[dict[str, list[float]]]:
    """
    Parse model output into a list of {"s", "a", "s_next"} each length raw_dim.
    Raises ValueError if structure is invalid.
    """
    blob = _extract_json_array(content)
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

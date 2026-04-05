"""
Generate ERGM transition data via Ollama (Gemma 4).

MicroGPT-style: small config dict at top, one script, writes .npz + .jsonl.

Requires: `ollama serve` with a Gemma 4 tag pulled, e.g. `ollama pull gemma4`

Run:
  python -m ergm.generate_training_data_ollama
  python -m ergm.generate_training_data_ollama --timeout 1200 --per-call 4
  python -m ergm.generate_training_data_ollama --model gemma4:4b --n 200 --out data/transitions

Env: OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT (seconds, default 600 in CONFIG).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error

import numpy as np

from ergm.ollama_client import ollama_chat
from ergm.parse_llm_transitions import parse_transition_array, transitions_to_numpy

# -----------------------------------------------------------------------------
# Micro-style defaults (edit here or override flags)
# -----------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


CONFIG = {
    "ollama_base_url": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "model": os.environ.get("OLLAMA_MODEL", "gemma4"),
    "raw_dim": 8,
    # Smaller batches finish faster on large local models (Gemma 4); use --per-call to tune.
    "transitions_per_call": 8,
    "target_total": 200,
    "out_prefix": "data/transitions_gemma4",
    "temperature_hint": "Use diverse but physically plausible numbers; avoid all zeros.",
    "timeout_s": _env_float("OLLAMA_TIMEOUT", 600.0),
}


SYSTEM_PROMPT = """You are a data generator for a geometric world model.
Output ONLY valid JSON: a single JSON array. No markdown, no explanation.

Each element must be an object with exactly three keys:
- "s": array of exactly {raw_dim} floating-point numbers (state before step)
- "a": array of exactly {raw_dim} floating-point numbers (action / control)
- "s_next": array of exactly {raw_dim} floating-point numbers (state after step)

CRITICAL: Every array entry must be a single numeric JSON literal (e.g. 0.11, -0.5, 2.0).
Never use expressions inside JSON: forbidden examples are 0.1+0.01, 1.0 + 0.2, 2*0.1 — compute the value yourself and write only the final number.

Interpret s as stacked physical channels (e.g. positions and velocities in arbitrary units).
s_next should be a plausible consequence of applying action a to state s for one discrete step
(smooth, bounded changes; no NaN or Infinity).

Generate {count} distinct transitions in one array."""


def build_user_prompt(raw_dim: int, count: int, batch_index: int, extra: str) -> str:
    return (
        f"Batch index {batch_index}. {extra}\n"
        f"Produce exactly {count} transition objects in one JSON array."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ERGM transitions via Ollama Gemma 4")
    parser.add_argument("--host", default=CONFIG["ollama_base_url"], help="Ollama base URL")
    parser.add_argument("--model", default=CONFIG["model"], help="Ollama model tag (e.g. gemma4, gemma4:4b)")
    parser.add_argument("--raw-dim", type=int, default=CONFIG["raw_dim"])
    parser.add_argument("--per-call", type=int, default=CONFIG["transitions_per_call"])
    parser.add_argument("--n", type=int, default=CONFIG["target_total"], help="Target number of transitions")
    parser.add_argument("--out", default=CONFIG["out_prefix"], help="Output path prefix (adds .npz and .jsonl)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=CONFIG["timeout_s"],
        help="HTTP timeout per Ollama request in seconds (env OLLAMA_TIMEOUT overrides default in CONFIG)",
    )
    args = parser.parse_args()

    base = args.host
    model = args.model
    raw_dim = args.raw_dim
    per = min(args.per_call, args.n)
    target = args.n
    out_prefix = args.out
    timeout_s = args.timeout

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    all_rows: list[dict[str, list[float]]] = []
    batch_idx = 0
    extras = [
        CONFIG["temperature_hint"],
        "Vary magnitudes across batches; include negative and positive values.",
        "Some batches: nearly linear step s_next ≈ s + 0.1 * a; still use exact floats in JSON.",
        "Include a few larger-velocity examples and some near-equilibrium small steps.",
    ]

    print(f"Ollama: {base}  model={model}  raw_dim={raw_dim}  target_n={target}  timeout_s={timeout_s}")
    while len(all_rows) < target:
        need = min(per, target - len(all_rows))
        sys_prompt = SYSTEM_PROMPT.format(raw_dim=raw_dim, count=need)
        extra = extras[batch_idx % len(extras)]
        user = build_user_prompt(raw_dim, need, batch_idx, extra)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        print(f"Request batch {batch_idx}: asking for {need} transitions...")
        t0 = time.perf_counter()
        try:
            content = ollama_chat(base, model, messages, timeout_s=timeout_s)
        except TimeoutError as e:
            print(f"{e}", file=sys.stderr)
            print(
                "Hints: export OLLAMA_TIMEOUT=1200  OR  python -m ergm.generate_training_data_ollama "
                "--timeout 1200 --per-call 4",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        except urllib.error.URLError as e:
            print(f"Connection failed ({e}). Is Ollama running? Try: ollama serve", file=sys.stderr)
            raise SystemExit(1) from e
        dt = time.perf_counter() - t0
        try:
            rows = parse_transition_array(content, raw_dim=raw_dim)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Parse error: {e}\n--- raw excerpt ---\n{content[:1200]}\n---", file=sys.stderr)
            raise SystemExit(2) from e

        if len(rows) > need:
            rows = rows[:need]
        elif len(rows) < need:
            print(f"Warning: model returned {len(rows)} rows, expected {need}; keeping partial batch.")

        all_rows.extend(rows)
        print(f"  got {len(rows)} in {dt:.1f}s (total {len(all_rows)})")
        batch_idx += 1
        if batch_idx > 200:
            print("Too many batches; abort.", file=sys.stderr)
            raise SystemExit(3)

    all_rows = all_rows[:target]
    s, a, sn = transitions_to_numpy(all_rows)

    npz_path = out_prefix + ".npz"
    jsonl_path = out_prefix + ".jsonl"

    np.savez(npz_path, s=s, a=a, s_next=sn)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {npz_path}  shapes s,a,s_next = {s.shape}, {a.shape}, {sn.shape}")
    print(f"Wrote {jsonl_path} ({len(all_rows)} lines)")


if __name__ == "__main__":
    main()

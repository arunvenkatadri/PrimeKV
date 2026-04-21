"""Autoresearch loop driver.

Karpathy-style: read the current ``experiment.py``, ask an agent to
propose a new version, evaluate it, keep it if it beats the best-so-
far score, revert otherwise. Loop forever (or for ``--rounds`` rounds).

Without an agent key (no ``--agent`` flag, or missing env vars), runs
in manual mode: evaluates the current ``experiment.py`` once and
prints the score. That's what the unit tests exercise.

Usage::

    # manual — just score the current experiment.py
    python -m autoresearch.run

    # with an agent — propose-evaluate-keep loop
    python -m autoresearch.run --agent anthropic --rounds 5

State is kept in ``autoresearch/state/``:

* ``best.py``       : best experiment.py so far
* ``best_score.txt``: its score
* ``history.jsonl`` : every attempt, for the log
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("autoresearch.run")


HERE = Path(__file__).parent
EXPERIMENT_PATH = HERE / "experiment.py"
STATE_DIR = HERE / "state"
BEST_PATH = STATE_DIR / "best.py"
BEST_SCORE_PATH = STATE_DIR / "best_score.txt"
HISTORY_PATH = STATE_DIR / "history.jsonl"
PROGRAM_PATH = HERE / "program.md"


DEFAULT_PROMPT = (
    "Large language models face a memory bottleneck when processing long "
    "contexts. One mitigation is a tiered cache that classifies each token "
    "by structural role and applies different retention policies per tier. "
    "Tokens flagged as anchors are kept in full precision; filler tokens "
    "are summarized. The question is whether this structural tiering "
    "outperforms attention-magnitude eviction on reasoning-heavy workloads."
)


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not BEST_PATH.exists():
        shutil.copy(EXPERIMENT_PATH, BEST_PATH)
    if not BEST_SCORE_PATH.exists():
        BEST_SCORE_PATH.write_text("-inf\n")


def read_best_score() -> float:
    try:
        val = BEST_SCORE_PATH.read_text().strip()
        return float(val)
    except (FileNotFoundError, ValueError):
        return float("-inf")


def write_best_score(score: float) -> None:
    BEST_SCORE_PATH.write_text(f"{score:.6f}\n")


def append_history(entry: dict) -> None:
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_experiment_module():
    """(Re)import ``autoresearch.experiment`` from disk."""
    if "autoresearch.experiment" in sys.modules:
        return importlib.reload(sys.modules["autoresearch.experiment"])
    return importlib.import_module("autoresearch.experiment")


def score_current_experiment(
    prompt: str = DEFAULT_PROMPT,
    model_name: str = "sshleifer/tiny-gpt2",
    device: str = "cpu",
) -> "autoresearch.prepare.EvalResult":  # noqa: F821
    from autoresearch import prepare

    model, tok = prepare.load_model(model_name=model_name, device=device)
    exp = load_experiment_module()
    num_layers = model.config.num_hidden_layers
    candidate = exp.build_candidate(num_layers=num_layers, device=device)
    return prepare.evaluate(candidate, model, tok, prompt=prompt, device=device)


# ---------------------------------------------------------------------------
# Agent backends
# ---------------------------------------------------------------------------


def _propose_with_anthropic(current_code: str, history: list[dict], program: str) -> Optional[str]:
    try:
        import anthropic  # type: ignore
    except ImportError:
        log.error("anthropic SDK not installed. `pip install anthropic`.")
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY not set.")
        return None

    client = anthropic.Anthropic(api_key=key)
    history_text = "\n".join(
        f"- round {h.get('round')}: score={h.get('score'):.3f} kept={h.get('kept')}"
        for h in history[-10:]
    ) or "(no history yet)"

    user_msg = (
        f"{program}\n\n"
        f"Here is the current experiment.py:\n\n```python\n{current_code}\n```\n\n"
        f"Recent attempts:\n{history_text}\n\n"
        "Propose a new experiment.py. Reply with ONLY the full file contents, "
        "no fences, no commentary."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
    )
    # Extract text from response
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def _propose_with_openai(current_code: str, history: list[dict], program: str) -> Optional[str]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        log.error("openai SDK not installed. `pip install openai`.")
        return None
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log.error("OPENAI_API_KEY not set.")
        return None
    client = OpenAI(api_key=key)

    history_text = "\n".join(
        f"- round {h.get('round')}: score={h.get('score'):.3f} kept={h.get('kept')}"
        for h in history[-10:]
    ) or "(no history yet)"
    user_msg = (
        f"{program}\n\n"
        f"Here is the current experiment.py:\n\n```python\n{current_code}\n```\n\n"
        f"Recent attempts:\n{history_text}\n\n"
        "Propose a new experiment.py. Reply with ONLY the full file contents, "
        "no fences, no commentary."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def propose(agent: str, current_code: str, history: list[dict]) -> Optional[str]:
    program = PROGRAM_PATH.read_text() if PROGRAM_PATH.exists() else ""
    if agent == "anthropic":
        return _propose_with_anthropic(current_code, history, program)
    if agent == "openai":
        return _propose_with_openai(current_code, history, program)
    raise ValueError(f"unknown agent: {agent}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _strip_code_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def run_loop(
    agent: Optional[str],
    rounds: int,
    prompt: str,
    model_name: str,
    device: str,
) -> None:
    ensure_state_dir()
    history: list[dict] = []

    # Score baseline (whatever's in experiment.py right now).
    log.info("scoring baseline experiment.py")
    result = score_current_experiment(prompt=prompt, model_name=model_name, device=device)
    best_score = result.score
    write_best_score(best_score)
    shutil.copy(EXPERIMENT_PATH, BEST_PATH)
    entry = {
        "round": 0,
        "score": best_score,
        "kept": True,
        "result": asdict(result),
        "ts": time.time(),
    }
    history.append(entry)
    append_history(entry)
    log.info("baseline score=%.4f", best_score)

    if agent is None:
        log.info("no agent specified — exiting after baseline.")
        return

    for r in range(1, rounds + 1):
        log.info("=== round %d/%d ===", r, rounds)
        current_code = EXPERIMENT_PATH.read_text()
        proposal = propose(agent, current_code, history)
        if not proposal:
            log.warning("agent returned no proposal; stopping.")
            break

        proposal = _strip_code_fences(proposal)
        prev = EXPERIMENT_PATH.read_text()
        try:
            EXPERIMENT_PATH.write_text(proposal)
            new_result = score_current_experiment(
                prompt=prompt, model_name=model_name, device=device
            )
            kept = new_result.score > best_score
            if kept:
                best_score = new_result.score
                write_best_score(best_score)
                shutil.copy(EXPERIMENT_PATH, BEST_PATH)
                log.info("kept: score=%.4f > best=%.4f", new_result.score, best_score)
            else:
                EXPERIMENT_PATH.write_text(prev)
                log.info("reverted: score=%.4f <= best=%.4f", new_result.score, best_score)
            entry = {
                "round": r,
                "score": new_result.score,
                "kept": kept,
                "result": asdict(new_result),
                "ts": time.time(),
            }
        except Exception as e:  # pragma: no cover — defensive
            EXPERIMENT_PATH.write_text(prev)
            log.warning("round %d crashed: %s", r, e)
            entry = {"round": r, "score": float("-inf"), "kept": False, "error": str(e), "ts": time.time()}
        history.append(entry)
        append_history(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    run_loop(args.agent, args.rounds, args.prompt, args.model, args.device)


if __name__ == "__main__":
    main()

"""PrimeKV Gradio web interface.

Interactive comparison of KV cache strategies:

* Enter a prompt and tweak decode length.
* Pick which caches to compare (full / H2O / StreamingLLM / uniform
  quant / PrimeKV).
* Tune per-backend hyperparameters.
* See a comparison table, per-cache metrics, tier distribution for
  PrimeKV, and a sample decode from each cache.

Launch:
    pip install -e ".[web]"
    python webui/app.py
    # or: python -m webui.app

Then open http://127.0.0.1:7860. This runs entirely on CPU by default
(the default model is GPT-2 124M); pick a smaller ``--decode-tokens``
for faster feedback loops.
"""

from __future__ import annotations

# Gradio renders matplotlib figures server-side — force Agg before any
# plt import or Gradio's capture won't get a non-interactive backend.
import matplotlib as _mpl
_mpl.use("Agg")

import argparse
import logging
import traceback
from typing import Any

import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    StreamingLLMCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier, Tier
from primekv.eval import Workload, run_comparison

log = logging.getLogger("primekv.webui")

# Module-level model cache so clicking "Run" doesn't re-download weights.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _load_model(name: str, device: str):
    key = f"{name}:{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    model.to(device)
    _MODEL_CACHE[key] = (tok, model)
    return tok, model


def _build_caches(
    selected: list[str],
    num_layers: int,
    h2o_cap: int,
    stream_window: int,
    primekv_cap: int,
    primekv_stride: int,
    primekv_anchor: int,
) -> dict:
    factories = {
        "full": lambda: FullCache(num_layers),
        "uniform_int8": lambda: UniformQuantCache(num_layers, bits=8),
        "uniform_int4": lambda: UniformQuantCache(num_layers, bits=4),
        "h2o": lambda: H2OCache(num_layers, capacity=int(h2o_cap)),
        "streamingllm": lambda: StreamingLLMCache(
            num_layers, num_sinks=4, window=int(stream_window)
        ),
        "primekv": lambda: PrimeKVCache(
            num_layers=num_layers,
            classifier=RuleBasedClassifier(
                anchor_prefix_len=int(primekv_anchor),
                semantic_stride=int(primekv_stride),
            ),
            max_entries_per_tier={Tier.SUPPORTING: int(primekv_cap)},
        ),
    }
    return {name: factories[name]() for name in selected if name in factories}


def _run(
    model_name: str,
    prompt: str,
    decode_tokens: float,
    max_length: float,
    selected: list[str],
    h2o_cap: float,
    stream_window: float,
    primekv_cap: float,
    primekv_stride: float,
    primekv_anchor: float,
):
    """Gradio callback. Returns (markdown, rows, tier_rows, generations_md)."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok, model = _load_model(model_name, device)

        num_layers = getattr(
            model.config,
            "num_hidden_layers",
            getattr(model.config, "n_layer", 12),
        )
        caches = _build_caches(
            selected,
            num_layers,
            int(h2o_cap),
            int(stream_window),
            int(primekv_cap),
            int(primekv_stride),
            int(primekv_anchor),
        )
        if not caches:
            return "**No caches selected.**", [], [], ""

        workload = Workload(
            prompt=prompt,
            decode_tokens=int(decode_tokens),
            max_length=int(max_length),
        )
        report = run_comparison(caches, workload, model, tok, device=device)

        # Convert dicts to list-of-lists for Gradio Dataframe rendering.
        row_dicts = report.as_rows()
        if row_dicts:
            cols = list(row_dicts[0].keys())
            rows = [[r[c] for c in cols] for r in row_dicts]
        else:
            rows = []

        # Tier distribution for PrimeKV, if it was in the run.
        tier_rows: list[list] = []
        for r in report.results:
            if r.name == "primekv" and "tier_distribution" in r.extra:
                for tier_name, count in r.extra["tier_distribution"].items():
                    tier_rows.append([tier_name, int(count)])
                break

        # Pretty-print each cache's greedy continuation.
        parts = []
        for r in report.results:
            if r.generated:
                parts.append(f"### {r.name}\n```\n{r.generated}\n```")
        generations_md = "\n\n".join(parts) or "_(no generations)_"

        return report.to_markdown(), rows, tier_rows, generations_md
    except Exception as e:  # pragma: no cover — surface errors in the UI
        log.exception("run failed")
        return f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```", [], [], ""


def _run_sweep(
    model_name: str,
    prompt: str,
    mode: str,
    capacities_str: str,
    lengths_str: str,
    ablate_axis: str,
    fixed_capacity: float,
    decode_tokens: float,
    max_length: float,
):
    """Gradio callback for the sweep tab. Returns (md, rows, plot, csv_text)."""
    try:
        from primekv.sweep import (
            plot_report,
            sweep_2d_tradeoff,
            sweep_ablate_primekv,
            sweep_pareto,
            sweep_vs_length,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok, model = _load_model(model_name, device)

        def _parse_ints(s: str, fallback: list[int]) -> list[int]:
            try:
                return [int(x) for x in s.replace(",", " ").split() if x.strip()]
            except Exception:
                return fallback

        if mode == "pareto":
            caps = _parse_ints(capacities_str, [4, 8, 16, 32, 64, 128, 256])
            report = sweep_pareto(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                capacities=caps,
                decode_tokens=int(decode_tokens),
                max_length=int(max_length),
                device=device,
            )
        elif mode == "vs_length":
            lengths = _parse_ints(lengths_str, [64, 128, 256, 512])
            report = sweep_vs_length(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                lengths=lengths,
                capacity=int(fixed_capacity),
                decode_tokens=int(decode_tokens),
                device=device,
            )
        elif mode == "2d":
            caps = _parse_ints(capacities_str, [8, 16, 32, 64])
            report = sweep_2d_tradeoff(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                eviction_caps=caps,
                precisions=["fp16", "int8", "int4"],
                decode_tokens=int(decode_tokens),
                max_length=int(max_length),
                device=device,
            )
        elif mode == "ablate":
            presets = {
                "anchor_prefix_len": [0, 4, 8, 16, 32, 64],
                "semantic_stride": [1, 2, 3, 5, 10],
                "supporting_cap": [4, 8, 16, 32, 64, 128],
                "enable_dynamic_reclassification": [False, True],
            }
            values = presets[ablate_axis]
            report = sweep_ablate_primekv(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                axis=ablate_axis,
                values=values,
                capacity=int(fixed_capacity),
                decode_tokens=int(decode_tokens),
                max_length=int(max_length),
                device=device,
            )
        else:
            return f"**Unknown sweep mode:** {mode}", [], None, ""

        # Build the rendered figure and a flat table.
        fig = plot_report(report)
        rows = []
        for p in report.points:
            rows.append(
                [
                    p.cache,
                    round(p.sweep_value, 3),
                    round(p.memory_bytes / (1024 * 1024), 3),
                    round(p.compression_ratio, 2),
                    round(p.perplexity, 3) if p.perplexity is not None else None,
                    round(p.prefill_ms, 2),
                    round(p.decode_ms, 2),
                ]
            )

        md = (
            f"**Mode:** `{report.mode}`  |  "
            f"**Axis:** `{report.axis_label}`  |  "
            f"**Points:** {len(report.points)}"
        )
        return md, rows, fig, report.to_csv()
    except Exception as e:  # pragma: no cover
        log.exception("sweep failed")
        return f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```", [], None, ""


def build_demo():
    import gradio as gr  # imported lazily so ``import webui.app`` without gradio still works

    with gr.Blocks(title="PrimeKV Comparison") as demo:
        gr.Markdown(
            "# PrimeKV\n"
            "Priority-Managed Inference KV Cache — interactive comparison harness.\n"
            "Use the **Compare** tab for a single run, or the **Sweep** tab for "
            "parameter sweeps with plots."
        )

        with gr.Tab("Compare"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_name = gr.Textbox(value="gpt2", label="HuggingFace model id")
                    prompt = gr.Textbox(
                        value=(
                            "System: you are a careful assistant.\n"
                            "User: Summarize the following paragraph. "
                            "The quick brown fox jumps over the lazy dog every afternoon, "
                            "rain or shine, ever since the summer of 1994.\n"
                            "Assistant:"
                        ),
                        lines=6,
                        label="Prompt",
                    )
                    decode_tokens = gr.Slider(1, 128, value=16, step=1, label="Decode tokens")
                    max_length = gr.Slider(16, 1024, value=128, step=16, label="Max prompt length")
                    selected = gr.CheckboxGroup(
                        choices=[
                            "full",
                            "uniform_int8",
                            "uniform_int4",
                            "h2o",
                            "streamingllm",
                            "primekv",
                        ],
                        value=["full", "uniform_int4", "h2o", "streamingllm", "primekv"],
                        label="Caches to compare",
                    )
                    with gr.Accordion("Cache hyperparameters", open=False):
                        h2o_cap = gr.Slider(4, 1024, value=256, step=4, label="H2O capacity")
                        stream_window = gr.Slider(
                            4, 1024, value=256, step=4, label="StreamingLLM window"
                        )
                        primekv_cap = gr.Slider(
                            4, 1024, value=256, step=4, label="PrimeKV Tier.SUPPORTING cap"
                        )
                        primekv_stride = gr.Slider(
                            1, 10, value=3, step=1, label="PrimeKV semantic stride"
                        )
                        primekv_anchor = gr.Slider(
                            0, 32, value=4, step=1, label="PrimeKV anchor prefix length"
                        )
                    run_btn = gr.Button("Run comparison", variant="primary")

                with gr.Column(scale=2):
                    md_out = gr.Markdown(label="Comparison (markdown)")
                    table = gr.Dataframe(
                        label="Metrics",
                        headers=[
                            "cache",
                            "memory_MB",
                            "ratio",
                            "ppl",
                            "prefill_ms",
                            "decode_ms",
                            "tok_per_s",
                        ],
                        interactive=False,
                        wrap=True,
                    )
                    tier_table = gr.Dataframe(
                        label="PrimeKV tier distribution",
                        headers=["Tier", "Tokens"],
                        interactive=False,
                    )
                    generations = gr.Markdown(label="Sample decodes")

            run_btn.click(
                _run,
                inputs=[
                    model_name,
                    prompt,
                    decode_tokens,
                    max_length,
                    selected,
                    h2o_cap,
                    stream_window,
                    primekv_cap,
                    primekv_stride,
                    primekv_anchor,
                ],
                outputs=[md_out, table, tier_table, generations],
            )

        with gr.Tab("Sweep"):
            gr.Markdown(
                "Run a parameter sweep and plot quality vs compression, vs "
                "prompt length, or as an ablation on one PrimeKV knob. "
                "Sweeps run the same model + prompt across many settings, so "
                "they take longer than single comparisons."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sweep_model = gr.Textbox(value="gpt2", label="HuggingFace model id")
                    sweep_prompt = gr.Textbox(
                        value=(
                            "The Eiffel Tower is a wrought-iron lattice tower on "
                            "the Champ de Mars in Paris, France. It is named "
                            "after the engineer Gustave Eiffel, whose company "
                            "designed and built the tower from 1887 to 1889."
                        ),
                        lines=5,
                        label="Prompt",
                    )
                    sweep_mode = gr.Radio(
                        choices=["pareto", "vs_length", "2d", "ablate"],
                        value="pareto",
                        label="Sweep mode ('2d' = eviction × quantization tradeoff)",
                    )
                    sweep_capacities = gr.Textbox(
                        value="4 8 16 32 64 128",
                        label="Capacities (space-separated) — for 'pareto'",
                    )
                    sweep_lengths = gr.Textbox(
                        value="64 128 256 512",
                        label="Prompt lengths (space-separated) — for 'vs_length'",
                    )
                    sweep_axis = gr.Dropdown(
                        choices=[
                            "anchor_prefix_len",
                            "semantic_stride",
                            "supporting_cap",
                            "enable_dynamic_reclassification",
                        ],
                        value="anchor_prefix_len",
                        label="Ablation axis — for 'ablate'",
                    )
                    sweep_fixed_cap = gr.Slider(
                        4, 1024, value=16, step=4,
                        label="Fixed capacity (for 'vs_length' and 'ablate')",
                    )
                    sweep_decode = gr.Slider(
                        1, 64, value=8, step=1, label="Decode tokens (per run)"
                    )
                    sweep_max_len = gr.Slider(
                        32, 1024, value=256, step=32, label="Max prompt length"
                    )
                    sweep_btn = gr.Button("Run sweep", variant="primary")

                with gr.Column(scale=2):
                    sweep_md = gr.Markdown()
                    sweep_plot = gr.Plot(label="Plot")
                    sweep_table = gr.Dataframe(
                        label="Sweep results",
                        headers=[
                            "cache",
                            "axis_value",
                            "memory_MB",
                            "ratio",
                            "ppl",
                            "prefill_ms",
                            "decode_ms",
                        ],
                        interactive=False,
                        wrap=True,
                    )
                    sweep_csv = gr.Textbox(
                        label="CSV (copy-paste friendly)",
                        lines=6,
                    )

            sweep_btn.click(
                _run_sweep,
                inputs=[
                    sweep_model,
                    sweep_prompt,
                    sweep_mode,
                    sweep_capacities,
                    sweep_lengths,
                    sweep_axis,
                    sweep_fixed_cap,
                    sweep_decode,
                    sweep_max_len,
                ],
                outputs=[sweep_md, sweep_table, sweep_plot, sweep_csv],
            )

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public gradio share link.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

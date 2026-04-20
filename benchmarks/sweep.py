"""Parameter sweep CLI.

Three modes:

    python benchmarks/sweep.py --pareto
    python benchmarks/sweep.py --vs-length
    python benchmarks/sweep.py --ablate anchor_prefix_len

Each writes ``runs/<ts>-sweep/results.{json,csv}`` and ``plot.png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from primekv.sweep import (
    SweepReport,
    plot_report,
    sweep_ablate_primekv,
    sweep_pareto,
    sweep_vs_length,
)

from benchmarks._common import dump_json, load_model, make_run_dir, pick_device


DEFAULT_PROMPT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
    "designed and built the tower from 1887 to 1889. Locally nicknamed La dame "
    "de fer, it was constructed as the centrepiece of the 1889 World's Fair and "
    "to crown the centennial anniversary of the French Revolution. Although "
    "initially criticised by some of France's leading artists and intellectuals "
    "for its design, it has since become a global cultural icon of France and "
    "one of the most recognisable structures in the world."
)


def _write_report(report: SweepReport, name: str) -> Path:
    out_dir = make_run_dir(name)
    dump_json(out_dir / "results.json", report.to_dict())
    (out_dir / "results.csv").write_text(report.to_csv())
    plot_report(report, output_path=str(out_dir / "plot.png"))
    print(f"wrote {out_dir}/results.{{json,csv}} and plot.png")
    return out_dir


def _print_summary(report: SweepReport) -> None:
    groups = report.by_cache()
    for name, pts in groups.items():
        print(f"\n{name}:")
        for p in pts:
            ppl_str = f"{p.perplexity:.3f}" if p.perplexity is not None else "  --"
            print(
                f"  {report.axis_label}={p.sweep_value:<8g}  "
                f"ratio={p.compression_ratio:>6.2f}x  "
                f"ppl={ppl_str}  "
                f"mem={p.memory_bytes/1024/1024:.2f}MB"
            )


def cmd_pareto(args, model, tokenizer, device: str) -> None:
    report = sweep_pareto(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        capacities=args.capacities,
        decode_tokens=args.decode_tokens,
        max_length=args.max_length,
        device=device,
        progress=print,
    )
    _print_summary(report)
    _write_report(report, "sweep-pareto")


def cmd_vs_length(args, model, tokenizer, device: str) -> None:
    report = sweep_vs_length(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        lengths=args.lengths,
        capacity=args.capacity,
        decode_tokens=args.decode_tokens,
        device=device,
        progress=print,
    )
    _print_summary(report)
    _write_report(report, "sweep-vs-length")


def cmd_ablate(args, model, tokenizer, device: str) -> None:
    axis = args.ablate
    if axis == "anchor_prefix_len":
        values = [0, 4, 8, 16, 32, 64]
    elif axis == "semantic_stride":
        values = [1, 2, 3, 5, 10]
    elif axis == "supporting_cap":
        values = [4, 8, 16, 32, 64, 128]
    elif axis == "enable_dynamic_reclassification":
        values = [False, True]
    else:
        raise SystemExit(f"unknown ablation axis: {axis}")
    report = sweep_ablate_primekv(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        axis=axis,
        values=values,
        capacity=args.capacity,
        decode_tokens=args.decode_tokens,
        max_length=args.max_length,
        device=device,
        progress=print,
    )
    _print_summary(report)
    _write_report(report, f"sweep-ablate-{axis}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--verbose", action="store_true")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pareto", action="store_true")
    mode.add_argument("--vs-length", action="store_true")
    mode.add_argument(
        "--ablate",
        choices=[
            "anchor_prefix_len",
            "semantic_stride",
            "supporting_cap",
            "enable_dynamic_reclassification",
        ],
    )

    # Sweep-specific knobs.
    ap.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
        help="capacity grid for --pareto",
    )
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="prompt length grid for --vs-length",
    )
    ap.add_argument("--capacity", type=int, default=16, help="fixed capacity for --vs-length/--ablate")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    device = pick_device()
    tok, model = load_model(args.model, device=device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.pareto:
        cmd_pareto(args, model, tok, device)
    elif args.vs_length:
        cmd_vs_length(args, model, tok, device)
    elif args.ablate:
        cmd_ablate(args, model, tok, device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GreenMind log analyzer — turns vLLM server logs (val1.log, val2.log, ...) into
a summary table and comparison charts for EXPERIMENTS.md and the pitch slides.

Usage:
    python3 analyze_logs.py val1.log val2.log            # summary + charts
    python3 analyze_logs.py val*.log --out charts        # choose output folder
    python3 analyze_logs.py val*.log --no-plot           # summary only (no matplotlib needed)

Reads only log files; never touches the GPU or the server.
"""
import argparse
import os
import re
import sys
from datetime import datetime

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STATS = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%"
    r"(?:, Prefix cache hit rate: ([\d.]+)%)?"
)
SPEC = re.compile(
    r"Mean acceptance length: ([\d.]+).*?Avg Draft acceptance rate: ([\d.]+)%"
)
ARGS = re.compile(r"non-default args: (\{.*\})")


def parse(path):
    stats, spec, config = [], [], None
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            if config is None:
                m = ARGS.search(line)
                if m:
                    config = m.group(1)
            ts = TS.search(line)
            if not ts:
                continue
            t = datetime.strptime("2026-" + ts.group(1), "%Y-%m-%d %H:%M:%S")
            m = STATS.search(line)
            if m:
                stats.append({
                    "t": t,
                    "prompt_tps": float(m.group(1)),
                    "gen_tps": float(m.group(2)),
                    "running": int(m.group(3)),
                    "waiting": int(m.group(4)),
                    "kv_pct": float(m.group(5)),
                    "prefix_hit_pct": float(m.group(6) or 0),
                })
                continue
            m = SPEC.search(line)
            if m:
                spec.append({"t": t, "accept_len": float(m.group(1)),
                             "accept_pct": float(m.group(2))})
    return stats, spec, config


def last_run(stats, gap_s=60):
    """Keep only the last busy stretch (the latest grader run).
    A log can contain health checks, curl tests and several runs; runs are
    separated by >= gap_s seconds with no activity."""
    busy = [s for s in stats if s["prompt_tps"] > 0 or s["gen_tps"] > 0 or s["running"] > 0]
    if not busy:
        return []
    segments, cur = [], [busy[0]]
    for s in busy[1:]:
        if (s["t"] - cur[-1]["t"]).total_seconds() > gap_s:
            segments.append(cur)
            cur = [s]
        else:
            cur.append(s)
    segments.append(cur)
    # The grader run is the longest segment; ties go to the latest.
    seg = max(reversed(segments), key=lambda g: (g[-1]["t"] - g[0]["t"]).total_seconds())
    start, end = seg[0]["t"], seg[-1]["t"]
    return [dict(s, sec=(s["t"] - start).total_seconds())
            for s in stats if start <= s["t"] <= end]


def summarize(name, run, spec):
    if not run:
        return {"log": name, "note": "no activity found"}
    start, end = run[0]["t"], run[-1]["t"]
    sp = [s for s in spec if start <= s["t"] <= end]
    total_prompt = sum(s["prompt_tps"] for s in run) * 10  # vLLM logs every 10 s
    total_gen = sum(s["gen_tps"] for s in run) * 10
    return {
        "log": name,
        "active_s": int((end - start).total_seconds()) + 10,
        "peak_prompt_tps": max(s["prompt_tps"] for s in run),
        "peak_gen_tps": max(s["gen_tps"] for s in run),
        "max_running": max(s["running"] for s in run),
        "max_waiting": max(s["waiting"] for s in run),
        "max_kv_pct": max(s["kv_pct"] for s in run),
        "prefix_hit_pct": run[-1]["prefix_hit_pct"],
        "approx_prompt_tok": int(total_prompt),
        "approx_gen_tok": int(total_gen),
        "mtp_accept_len": round(sum(s["accept_len"] for s in sp) / len(sp), 2) if sp else None,
        "mtp_accept_pct": round(sum(s["accept_pct"] for s in sp) / len(sp), 1) if sp else None,
    }


def print_table(rows):
    cols = ["log", "active_s", "peak_prompt_tps", "peak_gen_tps", "max_running",
            "max_waiting", "max_kv_pct", "prefix_hit_pct", "approx_gen_tok",
            "mtp_accept_len", "mtp_accept_pct"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + " |")


def plot(runs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping charts. Install with:\n"
              "  pip install --break-system-packages matplotlib", file=sys.stderr)
        return
    os.makedirs(out_dir, exist_ok=True)
    panels = [("prompt_tps", "Prompt (prefill) throughput, tok/s"),
              ("gen_tps", "Generation throughput, tok/s"),
              ("running", "Running requests"),
              ("waiting", "Waiting requests"),
              ("kv_pct", "GPU KV cache usage, %")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    for name, run in runs.items():
        if not run:
            continue
        x = [s["sec"] for s in run]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(x, [s[key] for s in run], marker=".", label=name)
            ax.set_title(title, fontsize=10, loc="left")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Seconds since run start")
    fig.suptitle("GreenMind — vLLM server behaviour per experiment", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "compare.png")
    fig.savefig(path, dpi=130)
    print(f"\nChart saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="vLLM log files, e.g. val1.log val2.log")
    ap.add_argument("--out", default="charts", help="output folder for charts (default: charts)")
    ap.add_argument("--no-plot", action="store_true", help="print summary only")
    a = ap.parse_args()

    rows, runs = [], {}
    for path in a.logs:
        name = os.path.splitext(os.path.basename(path))[0]
        stats, spec, config = parse(path)
        run = last_run(stats)
        runs[name] = run
        rows.append(summarize(name, run, spec))
        if config:
            print(f"{name} config: {config}")
    print()
    print_table(rows)
    if not a.no_plot:
        plot(runs, a.out)


if __name__ == "__main__":
    main()

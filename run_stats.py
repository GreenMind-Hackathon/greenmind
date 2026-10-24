#!/usr/bin/env python3
"""
GreenMind run stats: what happened to the requests in ONE graded run, from vLLM's
built-in /metrics counters (aggregate numbers only, no prompt or answer content).

Answers the question "why weren't all requests on time?":
  - how many requests hit the max_tokens limit (finished_reason=length): these can
    never meet the SLO (the grader requires "finished, not truncated"), and no
    serving setting changes that,
  - how many had slow first tokens or slow per-token speed (TTFT / TPOT histograms),
  - preemptions and aborted requests.

Usage on the VM (never send chat traffic during a graded run; reading /metrics is
a tiny local GET, but simplest is: snapshot right before Submit, then after it ends):
    python3 run_stats.py before          # after warmup.py, right before clicking Submit
    ... grader run ...
    python3 run_stats.py after           # when the grader says "done"
Snapshots are saved in ./metrics/ (before.prom, after.prom).
"""
import os
import re
import sys
import urllib.request

URL = os.environ.get("METRICS_URL", "http://localhost:8000/metrics")
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfINFnaN]+)')
LABEL = re.compile(r'(\w+)="([^"]*)"')


def fetch():
    req = urllib.request.Request(URL)
    if os.environ.get("KEY"):
        req.add_header("Authorization", "Bearer " + os.environ["KEY"])
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode()


def parse(text):
    """-> {(name, frozenset(labels)): value}, summed over model/engine labels."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if name.endswith("_created"):
            continue
        lab = {k: v for k, v in LABEL.findall(labels) if k not in ("model_name", "engine")}
        key = (name, frozenset(lab.items()))
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def delta(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def find(d, contains, suffix=""):
    return {k: v for k, v in d.items() if contains in k[0] and k[0].endswith(suffix)}


def hist(d, contains):
    """Cumulative histogram -> sorted [(le, count_le)], total."""
    buckets = find(d, contains, "_bucket")
    if not buckets:
        return None, None, None
    name = sorted({k[0] for k in buckets})[0]
    pts = []
    for (n, lab), v in buckets.items():
        if n != name:
            continue
        le = dict(lab).get("le")
        if le is None:
            continue
        pts.append((float("inf") if le in ("+Inf", "inf") else float(le), v))
    pts.sort()
    total = pts[-1][1] if pts else 0
    return name, pts, total


def over(pts, total, x):
    """count of observations > x (using the largest bucket edge <= x)."""
    le_x = [c for le, c in pts if le <= x + 1e-12]
    return total - (le_x[-1] if le_x else 0)


def report(d):
    print("Requests finished during the run, by reason:")
    fin = find(d, "request_success")
    if fin:
        for (n, lab), v in sorted(fin.items(), key=lambda kv: -kv[1]):
            reason = dict(lab).get("finished_reason", "?")
            if v:
                print(f"   {reason:<8} {int(v):>5}")
        length = sum(v for (n, lab), v in fin.items() if dict(lab).get("finished_reason") == "length")
        print(f"   -> {int(length)} hit the max_tokens limit (truncated): these always miss the SLO"
              " and are usually wrong answers too; no server setting fixes them.")
    else:
        print("   (no request_success counter found)")

    name, pts, total = hist(d, "time_to_first_token")
    if pts:
        print(f"\nTTFT ({name}), {int(total)} requests:")
        for x in (0.5, 1.0, 2.5, 5.0, 10.0):
            print(f"   > {x:>4}s : {int(over(pts, total, x)):>4}")
        print("   (the grader's TTFT limit is 0.5 s + 0.25 s per 1K prompt tokens, so long prompts may take longer)")

    for key in ("request_time_per_output_token", "time_per_output_token", "inter_token_latency"):
        name, pts, total = hist(d, key)
        if pts:
            per_req = "request_" in name
            print(f"\nTPOT ({name}; {'one value per request' if per_req else 'per token, not per request'}), "
                  f"{int(total)} observations:")
            for x in (0.025, 0.05, 0.1):
                print(f"   > {x * 1000:>4.0f} ms : {int(over(pts, total, x)):>6}")
            break

    name, pts, total = hist(d, "request_generation_tokens")
    if pts:
        print(f"\nOutput length ({name}), {int(total)} requests:")
        for x in (1000, 2000, 5000):
            print(f"   > {x:>5} tokens : {int(over(pts, total, x)):>4}")

    name, pts, total = hist(d, "request_params_max_tokens")
    if pts:
        print(f"\nRequested max_tokens ({name}), {int(total)} requests:")
        edges = [0, 200, 1000, 5000, 10000, float("inf")]
        for lo, hi in zip(edges, edges[1:]):
            n = over(pts, total, lo) - over(pts, total, hi) if hi != float("inf") else over(pts, total, lo)
            print(f"   {lo:>6.0f} < max_tokens <= {hi:<6.0f}: {int(n):>4}")
        print("   (warmup.py uses 128 and the Health check is tiny: requests <= 200 are probably not grader items)")

    name, pts, total = hist(d, "request_prompt_tokens")
    if pts:
        print(f"\nPrompt length ({name}), {int(total)} requests:")
        for x in (1000, 5000, 10000, 20000):
            print(f"   > {x:>5} tokens : {int(over(pts, total, x)):>4}")

    v = sum(find(d, "num_preemptions").values())
    print(f"\nPreemptions: {int(v)}   (a running request was evicted for lack of KV-cache memory and must be"
          " recomputed; anything above ~0 means memory ran out at peak)")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        print(__doc__)
        sys.exit(1)
    os.makedirs(DIR, exist_ok=True)
    text = fetch()
    path = os.path.join(DIR, sys.argv[1] + ".prom")
    with open(path, "w") as f:
        f.write(text)
    if sys.argv[1] == "before":
        print(f"Saved {path}. Now submit; run 'python3 run_stats.py after' when the grader is done.")
        return
    before_p = os.path.join(DIR, "before.prom")
    if not os.path.exists(before_p):
        sys.exit("No before.prom: run 'python3 run_stats.py before' before the next submission.")
    d = delta(parse(open(before_p).read()), parse(text))
    report(d)


if __name__ == "__main__":
    main()

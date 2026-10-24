#!/usr/bin/env python3
"""
GreenMind warm-up — run this AFTER every server restart and BEFORE submitting.

Why: vLLM compiles some Triton kernels the first time it sees a new shape
(the log shows "Triton kernel JIT compilation during inference ... latency spike").
If that happens inside a graded run, every in-flight request stalls and can miss
its TTFT/TPOT SLO. This script sends a burst of varied requests (short chat, Thai,
~6K- and ~20K-token prompts, tool calls, concurrent load) so those kernels are
compiled before the grader starts.

Usage (on the VM):
    KEY=<your api key> python3 warmup.py
    KEY=<key> python3 warmup.py --rounds 3 --url http://localhost:8000/v1

Uses only the Python standard library. Takes ~1-2 minutes.
Never run it while a grader submission is running.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

WORDS = ("energy efficiency inference server token prompt cache latency throughput "
         "model request answer document section report policy customer order data "
         "result value table chapter summary question context memory budget").split()


def filler(n_tokens, seed):
    """Roughly n_tokens of English-ish text. A random nonce at the start defeats
    the prefix cache so the prefill kernels really run."""
    rng = random.Random(seed)
    words = [f"[doc-{seed}-{rng.randint(0, 10**9)}]"]
    for i in range(int(n_tokens * 0.75)):
        words.append(rng.choice(WORDS))
        if i % 15 == 14:
            words[-1] += "."
    return " ".join(words)


WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def build_requests(round_no):
    s = round_no * 1000
    reqs = []
    for i in range(10):
        reqs.append(("chat", {"messages": [{"role": "user", "content":
                    f"In two sentences, explain idea #{s + i}: why batching helps GPU efficiency."}]}))
    for i in range(4):
        reqs.append(("thai", {"messages": [{"role": "user", "content":
                    f"ข้อ {s + i}: อธิบายสั้นๆ ว่าทำไมการประหยัดพลังงานจึงสำคัญสำหรับศูนย์ข้อมูล"}]}))
    for i in range(4):
        reqs.append(("rag_6k", {"messages": [{"role": "user", "content":
                    filler(6000, s + 100 + i) + "\n\nQuestion: what is the first bracketed id above?"}]}))
    for i in range(2):
        reqs.append(("long_20k", {"messages": [{"role": "user", "content":
                    filler(20000, s + 200 + i) + "\n\nSummarize the text above in one sentence."}]}))
    for i in range(3):
        reqs.append(("tool", {"messages": [{"role": "user", "content":
                    f"What is the weather in Bangkok? (request {s + i})"}],
                    "tools": WEATHER_TOOL, "tool_choice": "auto"}))
    # second turn of a tool episode (assistant tool call + tool result)
    reqs.append(("tool_turn2", {"messages": [
        {"role": "user", "content": f"What is the weather in Chiang Mai? (episode {s})"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{s}", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Chiang Mai"})}}]},
        {"role": "tool", "tool_call_id": f"call_{s}", "content": json.dumps({"temp_c": 31, "sky": "cloudy"})},
    ], "tools": WEATHER_TOOL, "tool_choice": "auto"}))
    return reqs


def post(url, key, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # stream=True: read the SSE stream to the end, like the grader does
        first = None
        t0 = time.time()
        for line in r:
            if first is None and line.startswith(b"data:"):
                first = time.time() - t0
        return first


def get_model(base, key):
    req = urllib.request.Request(base + "/models")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    key = os.environ.get("KEY", "")
    base = a.url.rstrip("/")

    try:
        model = get_model(base, key)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Server answered {e.code}. Is KEY set to your API key? ({e.reason})")
    except Exception as e:
        raise SystemExit(f"Cannot reach {base}: {e}. Is the server up?")
    print(f"Warming up {model} at {base}")

    for rnd in range(1, a.rounds + 1):
        reqs = build_requests(rnd)
        results = {}
        lock = threading.Lock()

        def worker(idx, kind, body):
            body = dict(body, model=model, max_tokens=a.max_tokens, temperature=0,
                        stream=True, stream_options={"include_usage": True})
            try:
                ttft = post(base + "/chat/completions", key, body, a.timeout)
                res = ("ok", ttft)
            except Exception as e:  # keep going; report at the end
                res = ("fail", str(e)[:120])
            with lock:
                results[idx] = (kind, res)

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i, k, b)) for i, (k, b) in enumerate(reqs)]
        for t in threads:
            t.start()
            time.sleep(0.05)  # small stagger, still overlapping
        for t in threads:
            t.join()
        took = time.time() - t0

        ok = sum(1 for _, (st, _) in results.values() if st == "ok")
        print(f"Round {rnd}: {ok}/{len(reqs)} ok in {took:.1f}s")
        by_kind = {}
        for kind, (st, val) in results.values():
            if st == "ok" and val is not None:
                by_kind.setdefault(kind, []).append(val)
        for kind, vals in sorted(by_kind.items()):
            print(f"   {kind:<10} max TTFT {max(vals):6.2f}s")
        for kind, (st, val) in results.values():
            if st != "ok":
                print(f"   FAILED {kind}: {val}")

    print("Warm-up done. Wait ~10 s for the GPU to go idle, then submit.")


if __name__ == "__main__":
    main()

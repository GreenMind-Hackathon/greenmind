#!/usr/bin/env python3
"""
GreenMind prefix-cache check — is vLLM actually reusing cached prompt prefixes?

The grader replays multi-turn tool-calling episodes (up to 15 turns). Every turn
resends the whole conversation, so most of each prompt should come from the
prefix cache. Our logs show a hit rate of only ~1%, so either the cache is not
working with this model/config, or the grader's prompts don't share prefixes.

This script tells the two apart:
  Test A  same ~4K-token document asked twice      -> 2nd request should be mostly cached
  Test B  a real 3-turn tool episode, built from the
          model's own replies (like the grader does) -> turns 2 and 3 should be mostly cached

It reads vLLM's /metrics counters before and after each request.
Run it on the VM after a restart (it also warms the server), NEVER during a grader run:
    KEY=<api key> python3 prefix_test.py
"""
import argparse
import json
import os
import random
import time
import urllib.request

WORDS = ("energy efficiency inference server token prompt cache latency throughput "
         "model request answer document section report policy customer order data "
         "result value table chapter summary question context memory budget").split()

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order", "description": "Look up an order by id. Returns status, items and total.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_policy", "description": "Return the refund policy text for a product category.",
        "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}}},
]


def text(n_tokens, seed):
    rng = random.Random(seed)
    w = [f"[ref-{seed}-{rng.randint(0, 10**9)}]"] + [rng.choice(WORDS) for _ in range(int(n_tokens * 0.75))]
    return " ".join(w)


class Client:
    def __init__(self, base, key):
        self.base, self.key = base.rstrip("/"), key
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base

    def _req(self, url, body=None):
        r = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                   method="GET" if body is None else "POST",
                                   headers={"Content-Type": "application/json"})
        if self.key:
            r.add_header("Authorization", f"Bearer {self.key}")
        return urllib.request.urlopen(r, timeout=600)

    def model(self):
        with self._req(self.base + "/models") as r:
            return json.load(r)["data"][0]["id"]

    def counters(self):
        hits = queries = None
        try:
            with self._req(self.root + "/metrics") as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("#") or "_created" in line:
                        continue
                    name, _, val = line.rpartition(" ")
                    if name.startswith("vllm:prefix_cache_hits"):
                        hits = (hits or 0) + float(val)
                    elif name.startswith("vllm:prefix_cache_queries"):
                        queries = (queries or 0) + float(val)
        except Exception:
            pass
        return hits, queries

    def chat(self, body):
        h0, q0 = self.counters()
        t0 = time.time()
        with self._req(self.base + "/chat/completions", body) as r:
            out = json.load(r)
        took = time.time() - t0
        time.sleep(0.5)
        h1, q1 = self.counters()
        cached = None if h0 is None or h1 is None else int(h1 - h0)
        queried = None if q0 is None or q1 is None else int(q1 - q0)
        return out, took, cached, queried


def report(label, out, took, cached, queried):
    pt = out.get("usage", {}).get("prompt_tokens")
    if cached is None:
        hit = "n/a (no prefix-cache metrics found)"
    else:
        pct = 100 * cached / queried if queried else 0
        hit = f"{cached:>6} of {queried:>6} prompt tokens from cache ({pct:4.1f}%)"
    print(f"  {label:<28} prompt={pt:>6}  {hit}  [{took:.1f}s]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    a = ap.parse_args()
    c = Client(a.url, os.environ.get("KEY", ""))
    model = c.model()
    seed = random.randint(0, 10**6)
    base = dict(model=model, temperature=0)

    print(f"Model: {model}\n\nTest A: same ~4K-token document, two different questions")
    doc = text(4000, seed)
    for i, q in enumerate(["What is the first bracketed reference?", "How many times does 'energy' appear, roughly?"]):
        out, took, cached, queried = c.chat(dict(base, max_tokens=64, messages=[
            {"role": "user", "content": doc + "\n\nQuestion: " + q}]))
        report(f"request {i + 1}", out, took, cached, queried)

    print("\nTest B: 3-turn tool episode built from the model's own replies")
    system = ("You are a customer-support agent for an online store. Use the tools to look up facts; "
              "never guess. Store handbook follows.\n" + text(2500, seed + 1))
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Order A-1043 arrived broken. Can I get a refund? Check the order and the policy."}]
    fake_results = [json.dumps({"order_id": "A-1043", "status": "delivered", "category": "kitchen", "total": 1290}),
                    json.dumps({"category": "kitchen", "policy": "Refund within 30 days if damaged on arrival."})]
    for turn in range(3):
        out, took, cached, queried = c.chat(dict(base, max_tokens=2048, messages=msgs,
                                                 tools=TOOLS, tool_choice="auto"))
        report(f"turn {turn + 1}", out, took, cached, queried)
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls or turn == 2:
            if not calls and turn < 2:
                print("  (model answered without a tool call; episode ended early)")
            break
        # Send back what an OpenAI client sends: content + tool_calls (no reasoning)
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            msgs.append({"role": "tool", "tool_call_id": call["id"],
                         "content": fake_results[min(turn, len(fake_results) - 1)]})

    print("\nHow to read this:")
    print("  Test A request 2 and Test B turns 2-3 should show most prompt tokens from cache.")
    print("  ~0% in Test A        -> prefix caching is not working in this server config.")
    print("  Test A OK, Test B ~0% -> caching works, but multi-turn history doesn't line up")
    print("                          (chat template / reasoning handling).")


if __name__ == "__main__":
    main()

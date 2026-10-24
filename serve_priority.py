#!/usr/bin/env python3
"""
GreenMind: SLO-aware admission order for vLLM ("shortest prompt first").

Why
---
The grader's TTFT target grows with prompt length: 0.5 s + 0.25 s per 1K prompt
tokens. A 300-token chat request must start in 0.575 s, a 20K-token document
gets 5.5 s. Under the Final's sustained peak the server is overloaded, and with
first-come-first-served order the cheap, tight-deadline requests queue behind
expensive, loose-deadline ones, so almost every peak request misses its SLO.

This wrapper gives each chat request a vLLM scheduling priority from its
estimated prompt size (smaller prompt = served first) and starts the normal
`vllm serve`. vLLM's own priority scheduler (--scheduling-policy priority) does
the rest. Nothing else changes: same model, same outputs, same sampling.
Every request is still answered; long ones just wait their turn, which their
looser deadline allows.

Rules: "Custom kernels, schedulers and serving code are welcome" (brief, rules
1/2). The priority uses only the prompt's size, which the scheduler sees anyway;
no prompt content is logged or stored.

Usage (inside the vllm/vllm-openai container; serve2.sh does this for you with PRIO=1):
    python3 serve_priority.py serve --model ... --scheduling-policy priority ...
    python3 serve_priority.py --check      # verify the patch points, then exit

Tuning (environment variables):
    GM_PRIO_BUCKET       tokens per priority level (default 256)
    GM_CHARS_PER_TOKEN   chars-per-token estimate (default 3.5)
"""
import importlib
import inspect
import json
import os
import pkgutil
import sys

BUCKET = max(1, int(os.environ.get("GM_PRIO_BUCKET", "256")))
CHARS_PER_TOKEN = float(os.environ.get("GM_CHARS_PER_TOKEN", "3.5"))
CANDIDATE_MODULES = [
    "vllm.entrypoints.openai.serving_chat",
    "vllm.entrypoints.openai.chat_completion.serving",
    "vllm.entrypoints.openai.chat_completion.serving_chat",
]
_logged = 0


def _plain(obj):
    """Turn pydantic objects / TypedDicts into JSON-able data."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(exclude_none=True)
        except Exception:
            pass
    return obj


def estimate_tokens(request):
    """Rough prompt size from the request body (messages + tool definitions)."""
    parts = []
    for m in getattr(request, "messages", None) or []:
        parts.append(_plain(m))
    tools = getattr(request, "tools", None) or []
    parts.extend(_plain(t) for t in tools)
    text = json.dumps(parts, ensure_ascii=False, default=str)
    return len(text) / CHARS_PER_TOKEN


def priority_for(request):
    return int(estimate_tokens(request) // BUCKET)


def _wrap(orig):
    def assign(request):
        global _logged
        try:
            if getattr(request, "priority", 0) in (0, None):
                p = priority_for(request)
                request.priority = p
                if _logged < 3:
                    _logged += 1
                    print(f"[greenmind] priority={p} (est. {p * BUCKET}-{(p + 1) * BUCKET} prompt tokens)",
                          file=sys.stderr, flush=True)
        except Exception as e:  # never break serving because of the patch
            print(f"[greenmind] priority not set: {e!r}", file=sys.stderr, flush=True)

    if inspect.isasyncgenfunction(orig):
        async def wrapped(self, request, *a, **k):
            assign(request)
            async for item in orig(self, request, *a, **k):
                yield item
    else:
        async def wrapped(self, request, *a, **k):
            assign(request)
            return await orig(self, request, *a, **k)
    wrapped._gm_patched = True
    wrapped.__wrapped__ = orig
    return wrapped


def find_serving_classes():
    found = []

    def scan(mod):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__ and hasattr(cls, "create_chat_completion"):
                found.append(cls)

    for name in CANDIDATE_MODULES:
        try:
            scan(importlib.import_module(name))
        except Exception:
            pass
    if not found:  # vLLM moved things around: search the OpenAI entrypoint package
        try:
            pkg = importlib.import_module("vllm.entrypoints.openai")
            for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                if "chat" not in info.name:
                    continue
                try:
                    scan(importlib.import_module(info.name))
                except Exception:
                    pass
        except Exception:
            pass
    return found


def apply_patch():
    classes = find_serving_classes()
    for cls in classes:
        if not getattr(cls.create_chat_completion, "_gm_patched", False):
            cls.create_chat_completion = _wrap(cls.create_chat_completion)
    return classes


def priority_enabled(argv):
    for i, a in enumerate(argv):
        if a == "--scheduling-policy" and i + 1 < len(argv) and argv[i + 1] == "priority":
            return True
        if a == "--scheduling-policy=priority":
            return True
    return False


def run_vllm_cli(args):
    sys.argv = ["vllm"] + args
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="console_scripts")
        ep = next((e for e in eps if e.name == "vllm"), None)
        if ep is not None:
            return ep.load()()
    except Exception:
        pass
    from vllm.entrypoints.cli.main import main
    return main()


def check():
    classes = apply_patch()
    if not classes:
        print("FAIL: could not find vLLM's chat serving class. Do not use PRIO=1.")
        sys.exit(1)
    print("Patched:", ", ".join(f"{c.__module__}.{c.__name__}" for c in classes))
    fields = None
    for name in ("vllm.entrypoints.openai.protocol", "vllm.entrypoints.openai.chat_completion.protocol"):
        try:
            req_cls = importlib.import_module(name).ChatCompletionRequest
            fields = getattr(req_cls, "model_fields", {})
            break
        except Exception:
            pass
    if fields is not None:
        print("ChatCompletionRequest has 'priority' field:", "priority" in fields)

    class Fake:
        messages = [{"role": "user", "content": "สวัสดี 2+2 เท่ากับเท่าไร"}]
        tools = None
        priority = 0
    short = Fake()
    long_ = Fake()
    long_.messages = [{"role": "user", "content": "word " * 20000}]
    print(f"Example priorities: short chat -> {priority_for(short)}, ~20K-token document -> {priority_for(long_)}")
    print("OK: start the server with --scheduling-policy priority (serve2.sh PRIO=1 does this).")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--check"]:
        check()
        sys.exit(0)
    if priority_enabled(args):
        classes = apply_patch()
        if classes:
            print(f"[greenmind] shortest-prompt-first priority active (bucket={BUCKET} tokens) on "
                  + ", ".join(c.__name__ for c in classes), file=sys.stderr, flush=True)
        else:
            print("[greenmind] WARNING: chat serving class not found; running WITHOUT priority patch",
                  file=sys.stderr, flush=True)
    else:
        print("[greenmind] --scheduling-policy priority not set; running WITHOUT priority patch",
              file=sys.stderr, flush=True)
    sys.exit(run_vllm_cli(args))

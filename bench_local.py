#!/usr/bin/env python3
"""Synthetic streaming load probe; run only when the grader is idle.

Reads the API key from KEY. This does not measure grader accuracy or score.
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


def percentile(values, p=0.95):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--requests', type=int, default=72)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['KEY']
    with urllib.request.urlopen(urllib.request.Request(a.url + '/models', headers=headers), timeout=30) as r:
        model = json.load(r)['data'][0]['id']

    def request(i):
        body = dict(model=model, messages=[{'role': 'user', 'content':
            f'Exercise {i}: Explain in detail how a GPU inference server can balance energy, latency, and throughput. Include examples and tradeoffs.'}],
            temperature=0, max_tokens=a.max_tokens, stream=True,
            stream_options={'include_usage': True})
        start = time.perf_counter()
        first = last = None
        tokens = None
        chunks = 0
        try:
            req = urllib.request.Request(a.url + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                for line in r:
                    if not line.startswith(b'data:') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[5:])
                    if event.get('usage'):
                        tokens = event['usage'].get('completion_tokens')
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                            last = time.perf_counter()
                            if first is None:
                                first = last
                            chunks += 1
            return dict(id=i, ok=first is not None, ttft_s=first-start if first else None,
                        tpot_s=(last-first)/(tokens-1) if first and tokens and tokens > 1 else None,
                        completion_tokens=tokens, chunks=chunks, elapsed_s=time.perf_counter()-start)
        except Exception as e:
            return dict(id=i, ok=False, error=str(e))

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.requests) as pool:
        results = list(pool.map(request, range(a.requests)))
    summary = dict(requests=a.requests, ok=sum(r['ok'] for r in results), elapsed_s=time.perf_counter()-start,
        p95_ttft_s=percentile([r['ttft_s'] for r in results if r.get('ttft_s') is not None]),
        p95_tpot_s=percentile([r['tpot_s'] for r in results if r.get('tpot_s') is not None]))
    with open(a.out, 'w') as f:
        json.dump(dict(summary=summary, results=results), f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

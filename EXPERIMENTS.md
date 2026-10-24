# Experiments — Team 14

Model: `CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp` (organizer-provided quantized variant, no weights of our own)
Engine: vLLM 0.32.0 (docker `vllm/vllm-openai:latest`) on 1x A100-SXM4 40GB
Baseline (organizer): BF16 `CMKL/MANGO1.5-Qwen3.5-9B`, default `vllm serve` → S = 0 by definition

Scores from the grader History page.

## Summary

| # | Submitted (UTC) | Config change vs previous | Track | S | A | R | s_E | s_T | s_L | η vs base | J/req | p95 TTFT | p95 TPOT | SLO met | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 Oct 04:01 | AWQ W4A16 + MTP 1 token, prefix caching, max-model-len 49152 | Validation | 42.6 | 1.00 | 1.013 | 0.41 | 0.43 | 0.47 | 1.57x | 236 | 2.85 s | 53 ms | 57/100 | 88.3% |
| 2 | 24 Oct 04:31 | + `--max-num-batched-tokens 8192`, + `--api-key` | Validation | **47.8** | 1.00 | 1.022 | 0.50 | 0.40 | 0.50 | 1.73x | 214 | 2.28 s | 58 ms | 57/100 | 89.2% |

## Run 1 — AWQ + MTP=1 (`serve.sh`)

Command:
```
--model CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp --max-model-len 49152 --gpu-memory-utilization 0.92
--enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```
Power limit: 400 W (default). Log: `val1.log`

Observed in server log:
- 340 chat requests, all HTTP 200, no errors
- Peak concurrency 52 running, max 3 waiting; KV cache peak 39.3% (595K-token pool)
- Peak throughput: prompt 6.5K tok/s, generation 1.4K tok/s
- MTP mean acceptance length 1.87 (of max 2.0), draft acceptance ~88%
- vLLM warned: `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` → token budget per step was small; motivated run 2

## Run 2 — + max-num-batched-tokens 8192 (`MTP=1 MBT=8192 KEY=... ./serve2.sh`)

Power limit: 400 W. Log: `val2.log`

Observed in server log:
- 338 requests, all HTTP 200, no errors
- Peak concurrency 66 running, 0 waiting (queueing gone); KV cache peak 50.2%
- Peak throughput: prompt 6.6K tok/s, generation 1.4K tok/s (same as run 1)
- MTP mean acceptance length 1.92, draft acceptance ~92%

Takeaway: S 42.6 → 47.8. The larger step budget removed queueing (p95 TTFT 2.85 → 2.28 s) and cut energy 236 → 214 J/request. Accuracy unchanged (A = 1, R > 1), so AWQ costs no accuracy here.

**Main problem in both runs: only 57/100 requests meet SLO.** p95 TPOT is 53–58 ms (> 50 ms limit), so many requests fail on TPOT during peak load. Those 43 requests burn energy but count as zero work, which caps both s_E and s_T.

## Next experiments (one variable at a time)

| Idea | Why | Expected effect |
|---|---|---|
| `--max-num-seqs` 32 / 48 | 66 concurrent decodes push p95 TPOT over 50 ms | Raise SLO-met above 57/100 → s_E and s_T both rise |
| MTP 2, 3 tokens | Acceptance ~92% at 1 token | Lower TPOT per token and fewer joules per token |
| Power limit 300 / 250 W | Only after TPOT has headroom: lower clocks raise TPOT | Further s_E gain if SLO-met holds |
| Accuracy check AWQ vs BF16 (Thai, reasoning) | Class factor can cut A sharply | Confirm A = 1 |

## What did not work / notes
- (add failed attempts here with the reason — judges ask for these)

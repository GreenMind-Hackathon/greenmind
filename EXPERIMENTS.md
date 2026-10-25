# Experiments — Team 14

**Best graded result: Final #15, S = 99.7** (A 1.00, R 1.046, s_E 1.00, s_T 0.98, s_L 1.00, 3.64x the baseline's on-time requests per joule, 164 J/request, 292/300 requests within SLO).

Hardware: 1x NVIDIA A100-SXM4 40GB, power limit 400 W (default, never changed). Engine: vLLM 0.32.0 (`vllm/vllm-openai:latest`).
Baseline (organizers): BF16 `CMKL/MANGO1.5-Qwen3.5-9B`, default `vllm serve`.
All scores are copied from the grader's History pages. Times are the grader's (UTC; Bangkok = UTC+7).

## Best configuration

Model: our own INT8 W8A8 quantization of the official BF16 weights (`quantize_int8.py`, data-free, no calibration data).

```bash
MODEL=/models/mango-w8a8 MTP=2 MBT=2048 LPT=1024 MAXSEQS=64 PRIO=1 KEY=$KEY ./serve2.sh
python3 warmup.py        # always, after every restart and before submitting
```

which runs (inside `vllm/vllm-openai`, through `serve_priority.py`):

```
vllm serve --model /models/mango-w8a8 --max-model-len 49152 --gpu-memory-utilization 0.92
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --max-num-seqs 64 --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024
  --scheduling-policy priority          # + serve_priority.py: smaller prompt = higher priority
  --api-key ...
```

Same configuration, seven graded Finals: 99.3, 93.3, 98.6, 99.2, 99.7, 99.4, 92.5 (median 99.2). The spread is run-to-run variation: with identical settings, R ranged 0.998-1.046, J/request 164-202 and p95 TTFT 1.06-2.27 s.

## Final runs (300 requests, graded)

| # | UTC | Model | Config (changes in bold) | S | A | R | s_E | s_T | s_L | η vs base | J/req | p95 TTFT | p95 TPOT | SLO met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 06:23 | AWQ | MTP 1, **MBT 8192** | 14.4 | 1.00 | 0.978 | 0.12 | 0.09 | 0.38 | 1.14x | 200 | 2.52 s | 87 ms | 112 |
| 2 | 07:13 | AWQ | **MTP 2, LPT 2048** | 20.5 | 0.98 | 0.980 | 0.21 | 0.08 | 0.50 | 1.26x | 183 | 1.98 s | 85 ms | 113 |
| 3 | 07:55 | AWQ | **+ PRIO** | 23.1 | 1.00 | 0.999 | 0.22 | 0.11 | 0.57 | 1.27x | 186 | 1.72 s | 81 ms | 116 |
| 4 | 08:14 | AWQ | **MBT 2048, LPT 512** | 0.0 | 1.00 | 0.993 | 0.00 | 0.05 | 0.37 | 0.86x | 206 | 7.70 s | 88 ms | 87 |
| 5 | 08:59 | AWQ | MBT 2048, LPT 2048, **MAXSEQS 64** | 44.0 | 1.00 | 1.003 | 0.45 | 0.38 | 0.47 | 1.64x | 205 | 3.67 s | 62 ms | 165 |
| 6 | 10:22 | AWQ | **MBT 1024, LPT 1024** | 65.1 | 1.00 | 0.989 | 0.73 | 0.65 | 0.46 | 2.24x | 199 | 4.77 s | 43 ms | 218 |
| 7 | 11:25 | **INT8** | same as #6 | 83.5 | 1.00 | 1.002 | 1.00 | 0.72 | 0.56 | 3.00x | 160 | 2.85 s | 30 ms | 235 |
| 8 | 11:58 | INT8 | **MAXSEQS 96** | 75.8 | 1.00 | 1.053 | 0.88 | 0.69 | 0.54 | 2.62x | 179 | 2.92 s | 31 ms | 229 |
| 9 | 12:28 | INT8 | **MBT 2048, LPT 2048**, MAXSEQS 64 | 89.2 | 1.00 | 1.009 | 0.97 | 0.82 | 0.76 | 2.90x | 179 | 1.57 s | 36 ms | 254 |
| 10 | 12:58 | INT8 | **LPT 1024** (= best config) | 99.3 | 1.00 | 1.011 | 1.00 | 0.96 | 1.00 | 3.27x | 179 | 1.06 s | 30 ms | 286 |
| 11 | 13:31 | INT8 | **MAXSEQS 72** | 98.8 | 1.00 | 1.000 | 1.00 | 0.94 | 1.00 | 3.16x | 185 | 1.57 s | 35 ms | 285 |
| 12 | 13:58 | INT8 | best config (repeat) | 93.3 | 1.00 | 1.022 | 0.93 | 0.89 | 1.00 | 2.77x | 202 | 1.63 s | 36 ms | 273 |
| 13 | 14:31 | INT8 | best config (repeat) | 98.6 | 1.00 | 1.035 | 1.00 | 0.93 | 1.00 | 3.20x | 180 | 1.49 s | 34 ms | 282 |
| 14 | 15:18 | INT8 | best config (repeat) | 99.2 | 1.00 | 0.998 | 1.00 | 0.96 | 1.00 | 3.28x | 179 | 1.54 s | 34 ms | 286 |
| **15** | **15:39** | **INT8** | **best config (repeat)** | **99.7** | 1.00 | 1.046 | 1.00 | 0.98 | 1.00 | 3.64x | 164 | 1.49 s | 33 ms | 292 |
| 16 | 16:19 | INT8 | best config (repeat) | 99.4 | 1.00 | 1.002 | 1.00 | 0.97 | 1.00 | 3.19x | 184 | 1.52 s | 33 ms | 287 |
| 17 | 17:28 | INT8 | best config (repeat), measured with `run_stats.py` | 92.5 | 1.00 | 1.037 | 0.96 | 0.83 | 0.92 | 2.87x | 187 | 2.27 s | 36 ms | 262 |
| 18 | 17:49 | INT8 | best config + **`--gpu-memory-utilization 0.95`** | 98.6 | 1.00 | 1.023 | 1.00 | 0.93 | 1.00 | 3.00x | 191 | 1.60 s | 33 ms | 279 |

Model: AWQ = `CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp` (organizers' variant); INT8 = `/models/mango-w8a8` (ours).
Shorthand: MBT = `--max-num-batched-tokens`, LPT = `--long-prefill-token-threshold`, MAXSEQS = `--max-num-seqs`, PRIO = shortest-prompt-first priority (`serve_priority.py`). Prefix caching on in every run. Unless changed in the row, settings carry over from the row above.

Logs: `final2.log` (#2), `final4.log` (#4), `final_mbt1024.log` (#6), `val_int8.log` (#7), `final_int8_seqs96.log` (#8), `final_int8_mbt2048.log` (#9), `final_int8_mbt2048_lpt1024.log` (#10), `final_int8_seqs72.log` (#11), `final_93.log` (#12), `final_986.log` (#13), `final_997.log` (#14 and #15), `final18_gmu095.log` (#18). Each file is the full container log, so it may also contain warm-up and validation traffic from the same server.

## Validation runs (100 requests, practice)

| # | UTC | Model | Config (changes in bold) | S | R | s_E | s_T | s_L | J/req | p95 TTFT | p95 TPOT | SLO met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 04:01 | AWQ | MTP 1, vLLM defaults (`serve.sh`) | 42.6 | 1.013 | 0.41 | 0.43 | 0.47 | 236 | 2.85 s | 53 ms | 57 |
| 2 | 04:31 | AWQ | **MBT 8192** | 47.8 | 1.022 | 0.50 | 0.40 | 0.50 | 214 | 2.28 s | 58 ms | 57 |
| 3 | 06:38 | AWQ | **LPT 2048**, warm-up added | 53.6 | 1.000 | 0.49 | 0.46 | 0.79 | 238 | 1.52 s | 55 ms | 63 |
| 4 | 06:59 | AWQ | **MTP 2** | 52.5 | 0.978 | 0.49 | 0.43 | 0.80 | 224 | 1.58 s | 51 ms | 59 |
| 5 | 07:51 | AWQ | **PRIO** | 52.4 | 1.000 | 0.49 | 0.43 | 0.76 | 219 | 1.67 s | 52 ms | 58 |
| 6 | 08:32 | AWQ | **MAXSEQS 64** | 56.1 | 1.041 | 0.53 | 0.48 | 0.79 | 222 | 1.63 s | 51 ms | 61 |
| 7 | 08:43 | AWQ | **MAXSEQS 48** | 51.7 | 1.041 | 0.49 | 0.48 | 0.67 | 225 | 2.22 s | 43 ms | 59 |
| 8 | 08:56 | AWQ | MAXSEQS 64, **MBT 2048** | 57.5 | 1.000 | 0.57 | 0.52 | 0.64 | 218 | 2.28 s | 50 ms | 63 |
| 9 | 10:19 | AWQ | **MBT 1024, LPT 1024** | 67.0 | 1.006 | 0.71 | 0.68 | 0.55 | 219 | 2.67 s | 39 ms | 74 |
| 10 | 10:40 | AWQ | **MAXSEQS 80** | 60.4 | 0.994 | 0.60 | 0.66 | 0.56 | 245 | 2.70 s | 39 ms | 73 |
| 11 | 10:50 | AWQ | **MAXSEQS 72** | 56.2 | 0.997 | 0.54 | 0.66 | 0.55 | 259 | 2.57 s | 39 ms | 72 |
| 12 | 11:22 | **INT8** | MBT 1024, LPT 1024, MAXSEQS 64 | 85.1 | 1.029 | 0.93 | 0.75 | 0.74 | 193 | 2.48 s | 24 ms | 82 |
| 13 | 11:54 | INT8 | **MAXSEQS 96** | 84.4 | 1.029 | 0.93 | 0.78 | 0.69 | 195 | 2.45 s | 24 ms | 83 |
| 14 | 12:09 | INT8 | **MBT 2048, LPT 2048**, MAXSEQS 64 | 92.4 | 1.016 | 0.94 | 0.85 | 0.96 | 204 | 1.14 s | 26 ms | 88 |
| 15 | 12:22 | INT8 | **MBT 3072, LPT 3072** | 88.0 | 1.032 | 0.86 | 0.85 | 0.96 | 221 | 0.99 s | 24 ms | 88 |
| 16 | 12:42 | INT8 | **MBT 2048, LPT 1024** (= best config) | 99.4 | 1.000 | 1.00 | 0.97 | 1.00 | 201 | 0.97 s | 23 ms | 97 |
| 17 | 13:14 | INT8 | **LPT 512** | 99.1 | 0.997 | 1.00 | 0.95 | 1.00 | 204 | 1.51 s | 24 ms | 95 |
| 18 | 13:26 | INT8 | LPT 1024, **MAXSEQS 72** | 99.4 | 1.003 | 1.00 | 0.97 | 1.00 | 198 | 1.06 s | 26 ms | 97 |
| 19 | 13:50 | INT8 | best config | 97.2 | 1.003 | 0.96 | 0.97 | 1.00 | 221 | 1.08 s | 25 ms | 98 |
| 20 | 14:12 | INT8 | best config | 99.7 | 1.038 | 1.00 | 0.98 | 1.00 | 187 | 1.18 s | 25 ms | 99 |
| 21 | 15:10 | **INT8-GPTQ** | best config | 87.8 | 1.019 | 0.82 | 0.94 | 1.00 | 248 | 1.12 s | 25 ms | 94 |

A was 1.00 in every validation run.

## What we learned, in order

**1. Validation scores did not predict Final scores.** The configuration of Validation #2 (47.8) scored 14.4 as Final #1. Validation's peak lasts a few seconds and the server absorbs it; the Final's peak lasts minutes, and the server was overloaded: running requests climbed to ~100, the KV cache reached 99%, p95 TPOT 81-88 ms. From then on we used Finals (2 per hour, only the best counts) to decide, and Validation only to check that a new configuration starts and keeps accuracy.

**2. Scheduler changes alone barely moved the AWQ Finals (14.4 → 23.1).** A larger step budget (MBT 8192) cut energy but not failures; shortest-prompt-first priority (`serve_priority.py` + vLLM's priority scheduler) helped a little (p95 TTFT 1.98 → 1.72 s). Priority can only reorder requests that are waiting, and without an admission limit almost nothing waited.

**3. Final #4 (S = 0) showed the real limit: memory.** MBT 2048 with LPT 512 let ~100 requests into the engine at once; the KV cache hit 99%, new requests queued, p95 TTFT rose to 7.7 s and on-time requests per joule fell below the baseline (0.86x), which scores 0. A 24-request synthetic warm-up burst had predicted the opposite: it does not reproduce a sustained peak.

**4. Admission control was the first large gain (23.1 → 44.0 → 65.1).** `--max-num-seqs 64` keeps the batch small enough to meet TPOT and keeps the KV cache away from its limit, and it creates a queue that the priority scheduler can order (short, tight-deadline prompts first). Smaller steps (MBT/LPT 1024) then brought p95 TPOT under 50 ms (43 ms).

**5. Our INT8 W8A8 model was the second large gain (65.1 → 83.5 with an identical scheduler).** The A100 has INT8 tensor cores but no FP8. The organizers' AWQ W4A16 model stores 4-bit weights but computes in 16-bit, so it does not speed up prompt processing. W8A8 runs the matrix multiplies on INT8 tensor cores (vLLM selected `CutlassInt8ScaledMMLinearKernel`). With the same scheduler settings (Final #6 → #7), p95 TPOT fell from 43 to 30 ms and energy per request from 199 to 160 J. Peak generation throughput was at most ~2.0K tokens/s in AWQ Finals and reached ~2.6K tokens/s with INT8. No accuracy loss: A = 1.00 and R 0.998-1.053 across all INT8 Finals.

**6. Re-tuning the scheduler for INT8 (83.5 → 99.3).** With a faster model, a slightly larger step (MBT 2048) and a smaller long-prompt slice (LPT 1024) worked best: long documents take at most 1,024 tokens per step, leaving room for short prompts every step. MAXSEQS 64 stayed best (72: 98.8, 96: 75.8).

**7. The energy term saturates.** From the grader's numbers, s_E = ln(η/η_base) / ln K with K ≈ 3 (Final #12: 2.77x → 0.93; Final #9: 2.90x → 0.97; 3.00x and above → 1.00). Our best runs reach 3.2-3.6x, so further energy savings cannot raise the score; the last 0.3 points are in s_T (0.98).

**8. The last points are not a memory or scheduling problem.** For Finals #17 and #18 we read vLLM's own `/metrics` counters before and after the run (`run_stats.py`; aggregate counts only, no prompt content):

| | Final #17 (92.5) | Final #18 (98.6, +5% KV memory) |
|---|---|---|
| Requests to the server (multi-turn items send several) | 673 | 663 |
| Finished normally / hit their `max_tokens` limit | 564 / 109 | 551 / 112 |
| Outputs longer than 5,000 tokens | 15 | 17 |
| TPOT above 50 ms | 3 | 2 |
| TTFT above 2.5 s (prompts above 20K tokens, whose limit is ≥ 5.5 s) | 10 (9) | 9 (9) |
| Preemptions counted by vLLM | 6,057 | 5,967 |

- Giving vLLM 5% more KV cache (562,623 → 592,687 tokens) did not change the preemption count, and the server log shows the KV cache at most 68% full with no waiting requests, so memory is not what limits us.
- Almost no request misses the TPOT limit, and the slow first tokens belong to the longest prompts, which have the most generous TTFT limit.
- What varies between identical runs is the answers themselves (R moves between 0.998 and 1.046), including which reasoning chains run until the token limit. A truncated request can never count as on time, and no serving setting changes that without changing the model's outputs, which the rules do not allow (sampling is fixed by the grader). We therefore stopped tuning at 99.7.

## What did not work (and why)

| Change | Result | Why |
|---|---|---|
| MBT 8192 alone (AWQ) | Validation SLO met unchanged (57 → 57) | Failures were not caused by the step budget; bigger steps only made each step longer |
| MBT 2048 + LPT 512, no admission limit (AWQ, Final #4) | S = 0.0 | ~100 requests in flight, KV cache 99%, queueing; TTFT 7.7 s |
| MAXSEQS 48 (Validation #7) | TPOT met (43 ms) but TTFT worse (2.22 s) | Queue too long |
| MAXSEQS 72 / 80 / 96 | Not better than 64 (Final #8 75.8, #11 98.8; Validation #10, #11, #13) | Larger batches push TPOT and memory up |
| MBT 3072 (INT8, Validation #15) | 88.0 vs 99.4 for MBT 2048 | Longer steps delay short prompts |
| MTP 2 vs 1 (Validation #3 vs #4) | Neutral on Validation; kept 2 for lower TPOT | Draft acceptance 2nd position ~74% under load |
| Prefix caching | Kept on, little effect | Works (~70% reuse in our `prefix_test.py`), but grader traffic shares few prefixes |
| INT8 with GPTQ calibration (`quantize_int8_gptq.py`, Validation #21) | 87.8 vs 97.2-99.7 for data-free INT8 | Higher J/request (248); one run only, no sign of benefit, not adopted |
| Synthetic warm-up burst as a benchmark | Predicted Final #4 wrongly | A 24-request burst is not a sustained peak; kept only for kernel warm-up |
| `--gpu-memory-utilization 0.95` (Final #18) | 98.6, preemptions unchanged | Memory is not the limit at MAXSEQS 64 |
| Disable image/video inputs (`--limit-mm-per-prompt '{"image":0,"video":0}'`, 24 Oct ~17:58) | Server failed at startup: `AttributeError: 'NoneType' object has no attribute 'size'` in `vllm/model_executor/models/qwen3_next.py`, then `RuntimeError: Engine core initialization failed`; never submitted | The cap of 64 had already lowered peak KV use to ~65%, so we reverted |

## Reproducibility and compliance

- Engine: open-source vLLM 0.32.0, one serving process on one A100. Our only code in the serving path is `serve_priority.py`, which sets each request's vLLM priority from its estimated prompt size; it does not log or store prompt content.
- Sampling, `max_tokens`, thinking mode and streaming are left exactly as the grader sends them.
- `warmup.py` sends synthetic prompts only, after each restart and before a submission, never during a graded run.
- GPU power limit and clocks were never changed (400 W default), so nothing differs between graded and ungraded time.
- `mango-w8a8` was made during the event from the official `CMKL/MANGO1.5-Qwen3.5-9B` weights with llm-compressor (open source): INT8 per-channel weights, dynamic per-token INT8 activations, no calibration data. `lm_head`, the vision tower, the MTP head and the small Gated-DeltaNet gate projections stay in BF16. Hugging Face: https://huggingface.co/PakornF/MANGO1.5-Qwen3.5-9B-W8A8-INT8-mtp.
- `mango-w8a8-gptq` (not used for the best score) used open calibration data only: `HuggingFaceH4/ultrachat_200k`, first 256 samples.
- `runs.log` records the settings of every launch (Bangkok time). Older entries do not record the model: every launch from 24 Oct 18:17 onward used `mango-w8a8`, except the 22:04 launch (`mango-w8a8-gptq`, Validation #21); earlier launches used the AWQ model.

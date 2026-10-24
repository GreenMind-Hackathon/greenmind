# GreenMind LLM Inference Optimization — Team 14

Serving `CMKL/MANGO1.5-Qwen3.5-9B` on 1x NVIDIA A100-SXM4 40GB with vLLM 0.32.0, optimized for on-time requests per joule.

**Best graded Final: S = 99.7** (A 1.00, R 1.046, s_E 1.00, s_T 0.98, s_L 1.00; 292/300 requests within SLO; 164 J per request; 3.64x the baseline's on-time requests per joule). Median of 7 repeated Finals with the same configuration: 99.2.

## What we did

1. **Our own INT8 W8A8 model.** The A100 has INT8 tensor cores but no FP8, and 4-bit weight-only models (AWQ W4A16) still compute in 16-bit. We quantized the official BF16 weights to INT8 weights + INT8 activations (data-free, llm-compressor), keeping the MTP head for speculative decoding.
   Weights: https://huggingface.co/PakornF/MANGO1.5-Qwen3.5-9B-W8A8-INT8-mtp
2. **Admission control.** `--max-num-seqs 64` keeps the batch small enough for the 50 ms TPOT limit and the KV cache below its limit during the sustained peak.
3. **Deadline-aware scheduling.** `serve_priority.py` gives shorter prompts (tighter TTFT limits) a higher vLLM priority; long prompts are fed in 1,024-token slices so they never block short ones.
4. **MTP speculative decoding** with 2 draft tokens (~2.7 tokens per step).

The full story, every graded run (18 Finals, 21 Validations) and everything that did not work: [EXPERIMENTS.md](EXPERIMENTS.md).

## Reproduce the best configuration

On the serving VM (Docker + NVIDIA runtime; model downloaded to `~/models/mango-w8a8`, or use the Hugging Face id):

```bash
export KEY=<api key>
MODEL=/models/mango-w8a8 MTP=2 MBT=2048 LPT=1024 MAXSEQS=64 PRIO=1 KEY=$KEY ./serve2.sh
docker logs -f vllm          # wait for "Application startup complete"
python3 warmup.py            # compile kernels before the first graded request
```

This runs `vllm/vllm-openai:latest` (vLLM 0.32.0) with:

```
--model /models/mango-w8a8 --max-model-len 49152 --gpu-memory-utilization 0.92
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
--max-num-seqs 64 --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024
--scheduling-policy priority   (started through serve_priority.py)
```

GPU power limit and clocks: default (400 W), never changed.

To rebuild the INT8 model from the official weights (CPU, ~2 min plus download): `python3 quantize_int8.py --out ~/models/mango-w8a8`.

## Repository map

| File | Purpose |
|---|---|
| `serve2.sh` | Launch script; every setting is an environment variable (documented at the top) |
| `serve_priority.py` | Wrapper around `vllm serve`: smaller prompt = higher scheduling priority |
| `warmup.py` | Synthetic warm-up burst after each restart (compiles kernels; also prints TTFT/TPOT per request type) |
| `quantize_int8.py` | Builds the INT8 W8A8 model used for our best score |
| `quantize_int8_gptq.py` | GPTQ-calibrated INT8 variant (tested, not adopted) |
| `run_stats.py` | Reads vLLM's `/metrics` before/after a run: truncations, TTFT/TPOT histograms, preemptions |
| `analyze_logs.py` | Summary table and charts from vLLM server logs |
| `prefix_test.py` | Checks whether prefix caching reuses prompt tokens |
| `EXPERIMENTS.md` | Every run with its configuration and grader scores, lessons, failures |
| `runs.log` | Settings and power limit of every server launch |
| `val*.log`, `final*.log` | Full vLLM container logs of the runs listed in `EXPERIMENTS.md` |
| `serve.sh` | First launch script (Validation #1) |

## Compliance

- Open-source engine (vLLM), one serving process on one GPU; our only serving code is `serve_priority.py`, which sets a priority from the prompt's size and never logs or stores prompt content.
- Sampling, `max_tokens`, thinking mode and streaming are untouched; requests are served as the grader sends them.
- `warmup.py` uses synthetic prompts only and is never run during a graded run.
- The INT8 model was made during the event from the official `CMKL/MANGO1.5-Qwen3.5-9B` weights with open-source llm-compressor and no calibration data; the GPTQ variant (not used for our score) used open data only (`HuggingFaceH4/ultrachat_200k`).

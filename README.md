# GreenMind LLM Inference Optimization — Team 14
Serving CMKL/MANGO1.5-Qwen3.5-9B on 1x A100 40GB with vLLM.
- `serve2.sh` — launch script (settings via env vars, e.g. `MTP=1 MBT=8192 KEY=... ./serve2.sh`)
- `EXPERIMENTS.md` — every run, its config and grader scores, including what didn't work
- `val*.log` — vLLM server logs for each validation run
- `runs.log` — exact settings and power limit of each launch

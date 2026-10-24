#!/bin/bash
docker rm -f vllm 2>/dev/null
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 49152 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

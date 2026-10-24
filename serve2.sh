#!/bin/bash
# Usage example:  MTP=0 ./serve2.sh     |   MTP=2 MBT=8192 KEY=abc ./serve2.sh
MTP=${MTP:-1}
ARGS=(--model CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp
      --host 0.0.0.0 --port 8000
      --max-model-len 49152 --gpu-memory-utilization 0.92
      --enable-prefix-caching
      --reasoning-parser qwen3
      --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "$MTP" -gt 0 ]   && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
[ -n "$MAXSEQS" ]  && ARGS+=(--max-num-seqs "$MAXSEQS")
[ -n "$MBT" ]      && ARGS+=(--max-num-batched-tokens "$MBT")
[ -n "$KEY" ]      && ARGS+=(--api-key "$KEY")
docker rm -f vllm 2>/dev/null
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest "${ARGS[@]}"
echo "$(date '+%F %T')  MTP=$MTP MAXSEQS=${MAXSEQS:-default} MBT=${MBT:-default}  PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" >> runs.log

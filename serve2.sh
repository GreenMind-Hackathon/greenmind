#!/bin/bash
# GreenMind launch script. Settings come from environment variables:
#   MTP      number of MTP draft tokens (0 = off)          default 1
#   MBT      --max-num-batched-tokens (prefill budget)      default: vLLM's
#   LPT      --long-prefill-token-threshold                 default: off
#   MAXSEQS  --max-num-seqs                                 default: vLLM's
#   KEY      API key                                        default: none
# Example:  MTP=1 MBT=8192 LPT=2048 KEY=$KEY ./serve2.sh
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
[ -n "$LPT" ]      && ARGS+=(--long-prefill-token-threshold "$LPT")
[ -n "$KEY" ]      && ARGS+=(--api-key "$KEY")

# Keep compile caches on the VM disk so restarts skip most of the ~2-3 min
# torch.compile / Triton warm-up (they were lost every time the container was recreated).
mkdir -p ~/.cache/huggingface ~/.cache/vllm ~/.cache/triton ~/.cache/torchinductor

docker rm -f vllm 2>/dev/null
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/triton:/root/.triton \
  -v ~/.cache/torchinductor:/tmp/torchinductor_root \
  vllm/vllm-openai:latest "${ARGS[@]}"

echo "$(date '+%F %T')  MTP=$MTP MBT=${MBT:-default} LPT=${LPT:-off} MAXSEQS=${MAXSEQS:-default}  PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)  SMCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader)" >> runs.log
echo "Started. Watch with: docker logs -f vllm   (wait for 'Application startup complete')"

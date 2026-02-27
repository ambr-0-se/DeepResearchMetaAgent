#!/bin/bash
# vLLM server startup script for Qwen3-VL-4B-Instruct
# Run with: ./start_vllm.sh or nohup ./start_vllm.sh > vllm_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 4}'

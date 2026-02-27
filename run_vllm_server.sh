#!/bin/bash
#SBATCH --job-name=vllm-qwen3vl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_4080:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_server_%j.out
#SBATCH --error=logs/vllm_server_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting vLLM server with Qwen3-VL-4B-Instruct"
echo "========================================"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dra

mkdir -p logs

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

cd /userhome/cs2/ambr0se/DeepResearchMetaAgent

echo "Starting vLLM server on port 8000..."
python -m vllm.entrypoints.openai.api_server \
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

echo "vLLM server stopped."

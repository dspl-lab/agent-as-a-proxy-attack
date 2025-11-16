#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

ALIGNMENTCHECK_LLMS=(
  Qwen/Qwen2.5-7B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  meta-llama/Llama-3.1-8B-Instruct
)

NUM_SAMPLES_LIST=(1 4 16)

MODES=("alignmentcheck")

for NUM_SAMPLES in "${NUM_SAMPLES_LIST[@]}"; do
  for ALIGNMENTCHECK_LLM in "${ALIGNMENTCHECK_LLMS[@]}"; do
    for MODE in "${MODES[@]}"; do
      echo "[METADATA] time=$(date -u +"%Y-%m-%dT%H:%M:%SZ") host=$(hostname) llm=$ALIGNMENTCHECK_LLM num_samples=$NUM_SAMPLES mode=$MODE"

      python3 -m python_scripts.run_universal_attacks \
        --num_training_samples "$NUM_SAMPLES" \
        --mode "$MODE" \
        --alignmentcheck_llm "$ALIGNMENTCHECK_LLM" \
        --agentdojo_path "./externals/agentdojo/reruns/optim_str" \
        --agentdojo_llms  gpt-4o-2024-05-13
    done
  done
done

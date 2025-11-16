#!/usr/bin/env bash
set -a
source "$(dirname "$0")/../.env"
CUDA_VISIBLE_DEVICES=7
set +a
REPL_MODES=("extractor")
KS=("1" "4" "16")
SUITE="slack"
SUITE_ARG=()
if [[ -n "$SUITE" ]];
then SUITE_ARG=(--suite "$SUITE")
fi
MONITOR_MODEL="hf/mistralai/mistral-7b-instruct-v0.3"
AGENTDOJO_PATH="../agentdojo/reruns/optim_str"
PROMPT_PATH="./src/cot_red_handed/monitor/prompts"

cd externals/aether-cot-monitoring

load_optim_strings() {
  OPTIM_STRINGS=()

  if [[ "$MONITOR_MODEL" == "hf/meta-llama/Llama-3.1-8B-Instruct" ]]; then
    if [[ "$K" == "1" ]]; then
      OPTIM_STRINGS=(
            "<id>" "<optim_str>"
            "<id>" "<optim_str>"
      )
    elif [[ "$K" == "4" ]]; then
      OPTIM_STRINGS=(
            "<id>" "<optim_str>"
      )
    elif [[ "$K" == "16" ]]; then
      OPTIM_STRINGS=(
            "<id>" "<optim_str>"
            "<id>" "<optim_str>"
      )
    else
      echo "Unknown K: $K" >&2
      return 1
    fi

  else
    echo "Unknown model: $MONITOR_MODEL" >&2
    return 1
  fi
}

for REPL_MODE in "${REPL_MODES[@]}"; do
  for K in "${KS[@]}"; do

    load_optim_strings || exit 1

    for ((i = 0; i < ${#OPTIM_STRINGS[@]}; i+=2)); do
      RESULTS_PATH="../../results/aether-cot-monitoring/${MONITOR_MODEL#hf/}/optim_str/${REPL_MODE}/${SUITE}/k_${K}/${OPTIM_STRINGS[i]}"

      EXTRACT_OPTIM_ARG=()
      [[ "$REPL_MODE" == "extractor" ]] && \
        EXTRACT_OPTIM_ARG=(--optim-str "${OPTIM_STRINGS[i+1]}")

      EVAL_OPTIM_ARG=()
      [[ "$REPL_MODE" == "evaluator" ]] && \
        EVAL_OPTIM_ARG=(--optim-str "${OPTIM_STRINGS[i+1]}")

      uv run src/cot_red_handed/monitor/run_monitor.py \
        --prompt-id 1 \
        --prompt-path "$PROMPT_PATH" \
        --dataset-path "$AGENTDOJO_PATH" \
        --log-dir "$RESULTS_PATH" \
        --monitor-model "$MONITOR_MODEL" \
        --max-connections 1 \
        --max-sandboxes 1 \
        --max-tasks 1 \
        "${SUITE_ARG[@]}" \
        "${EXTRACT_OPTIM_ARG[@]}"

      uv run src/cot_red_handed/monitor/run_monitor.py \
        --prompt-id 1 \
        --prompt-path "$PROMPT_PATH" \
        --dataset-path "$AGENTDOJO_PATH" \
        --log-dir "$RESULTS_PATH" \
        --hierarchical \
        --monitor-model "$MONITOR_MODEL" \
        --max-connections 1 \
        --max-sandboxes 1 \
        --max-tasks 1 \
        "${SUITE_ARG[@]}" \
        "${EVAL_OPTIM_ARG[@]}"
    done

  done
done

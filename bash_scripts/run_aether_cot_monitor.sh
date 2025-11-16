# original - original tasks
# optim_str - {optim_str}
# test - {k=1}

set -a
source "$(dirname "$0")/../.env"
CUDA_VISIBLE_DEVICES=3
set +a

SUITE="slack"   # or "workspace", or unset
MONITOR_MODEL="hf/mistralai/mistral-7b-instruct-v0.3"
AGENTDOJO_PATHS=(
    "../agentdojo/reruns/original"
    "../agentdojo/reruns/optim_str"
)
PROMPT_PATH="./src/cot_red_handed/monitor/prompts"

cd externals/aether-cot-monitoring

SUITE_ARG=()
if [[ -n "$SUITE" ]]; then
    SUITE_ARG=(--suite "$SUITE")
fi

RESULTS_PATH="../../results/aether-cot-monitoring/${MONITOR_MODEL#hf/}/optim_str/${SUITE}"
for AGENTDOJO_PATH in "${AGENTDOJO_PATHS[@]}"; do
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
        "${SUITE_ARG[@]}"
done

#!/bin/bash
# You can specify which LLMs to us as AlignmentCheck monitor
# Which LLM to choose as the agent
# If you choose an agent, make sure you have run AgentDojo
# with that agent before running this script
# You only need "alignmentcheck" and "promptguard_toolcheck_only_alignmentcheck_normal" modes
# This first one is just AlignmentCheck
# The second one is PromptGuard + AlignmentCheck
# If you want to play around with different LlamaFirewall configuration
# You can define new modes in python_scripts/utils.py and use them

export CUDA_VISIBLE_DEVICES=6
ALIGNMENTCHECK_MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "google/gemma-3-1b-it"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "lmsys/vicuna-7b-v1.5"
    # "CohereLabs/c4ai-command-r7b-12-2024"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-8B-FP8"
    # "google/gemma-2-9b-it"
    # "Qwen/Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # "google/gemma-3-12b-it"
)


AGENT_MODELS=(
    "gpt-4o-2024-05-13"
    "google_gemini-2.5-pro"
    )

MODES=(
    "alignmentcheck" 
    "promptguard_toolcheck_only_alignmentcheck_normal"
    )

AGENTDOJO_PATHS=(
    "externals/agentdojo/reruns/original"
    "externals/agentdojo/reruns/optim_str"
)


for MODE in "${MODES[@]}"; do
    for AGENTDOJO_PATH in "${AGENTDOJO_PATHS[@]}"; do
        for ALIGN_MODEL in "${ALIGNMENT_MODELS[@]}"; do
            for AGENT_MODEL in "${AGENT_MODELS[@]}"; do
                echo "Running $MODE | Model: $ALIGN_MODEL"
                
                python3 -m python_scripts.run_alignmentcheck \
                    --alignmentcheck_llm "$ALIGN_MODEL" \
                    --mode "$MODE" \
                    --security true \
                    --agentdojo_path "$AGENTDOJO_PATH" \
                    --agentdojo_llms "$AGENT_MODEL"
            done
        done
    done
done

echo "All checks completed."

# YOU CAN SKIP THIS SINCE WE HAVE ALREADY UPLOADED ALL RESULTS TO THE REPO

source envs/inference/bin/activate

python -m agentdojo.scripts.benchmark \
    --model GPT_4O_2024_05_13 \
    --provider openrouter \
    --logdir externals/agentdojo/reruns/original \
    --benchmark-version v1.2.1 \
    --attack important_instructions \
    --adapted False \
    # -s banking \ 

python -m agentdojo.scripts.benchmark \
    --model GPT_4O_2024_05_13 \
    --provider openrouter \
    --logdir externals/agentdojo/reruns/original \
    --benchmark-version v1.2.1 \
    --attack important_instructions \
    --adapted True \
    --optim_str "{optim_str}" \
    # -s banking \ # If you don't specify the suite, all suites will be run

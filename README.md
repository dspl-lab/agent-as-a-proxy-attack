# Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks

This repo contains experiments for the paper *Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks*. [![arXiv](https://img.shields.io/badge/arXiv-2602.05066-red.svg)](https://arxiv.org/abs/2602.05066)

This is a confusing problem to solve. As a result, repo may look confusing as well. We suggest reading through this very brief README before using the repo. We welcome all issues and pull requests.

## Set-up
Create an .env file and add the following variables:
```txt
HF_TOKEN="myhftoken"
OPENROUTER_API_KEY="myopenroutertoken"
OPENAI_API_KEY="myopenaitoken"
```
You can also skip OpenAI token and use OpenRouter for everything.

Next, we need to create two virtual environments, one for attack and one for inference.

> [!WARNING]
> We need two virtual environments because because both `nanogcg` and `AdaptiveAttackAgent` engines rely on an outdated prefix cache API from `transformers<=4.45.0`. We use one `attack` environment to use either of these engines to optimize attacks, but if you want to evaluate LLMs that require `transformers>4.45.0`, you will have to create another `inference` environment.


```sh

git clone --recursive https://github.com/ceferisbarov/agent-as-a-proxy-attack

mkdir envs

uv venv envs/attack
uv venv envs/inference

source envs/attack/activate
uv pip install -r requirements_attack.txt

source envs/inference/activate
uv pip install -r requirements_inference.txt
```

## Run monitors
Before we can optimize any attack strings, we need to have baseline results.

First, we need to run AgentDojo tasks. You would do it via:
```sh
./bash_scripts/agentdojo.sh
```
But you can skip this since we have already uploaded the results to the repo. You can access them at `externals/agentdojo/reruns/original` and `externals/agentdojo/reruns/optim_str`, respectively.

Next, we can run the monitors.

Run AlignmentCheck and extract-and-evaluate monitors with these scripts:

```sh
bash_scripts/run_alignmentcheck.sh
bash_scripts/run_aether_cot_monitor.sh
```

Again, we are in the process of uploading these logs and if they are not available, you can simply request them from us by opening an issue instead of running them yourselves.

## Generate targets
We also need to generate targets for AlignmentCheck (see Appendix D for more details). We can use `python3 -m python_scripts.generate_targets` to do this. We used targets generated for `Qwen/Qwen2.5-7B-Instruct` for all models, but you can generate different targets for different models as well.
  

Generating targets are not necessary for Extract-and-Evaluate monitor, because their output is in `<answer>SCORE</answer>` format, where SCORE is a number between 0 and 100. We simply use `<answer>0</answer>` as the target for all cases.

## Optimize attack strings

Now that we have the logs and the intended targets (AlignmentCheck outputs), we can optimize our attack strings to elicit these targets. You can use the following scripts to generate attacks:
```sh
CUDA_VISIBLE_DEVICES=4 python3 -m python_scripts.generate_gcg_attacks.py
CUDA_VISIBLE_DEVICES=4 python3 -m python_scripts.generate_multigcg_attacks.py
```
GCG attacks are intended for Extract-and-Evaluate and AlignmentCheck. MultiGCG attacks are necessary to bypass PromptGuard + AlignmentCheck at the same time.

## Run attacks
After optimizing attack strings, we need to use them against the monitors. You can use the following scripts to do that:

```sh
run_universal_attacks_aether.sh
run_universal_attacks_alignmentcheck.sh
```
We have removing attack strings from the repo, but they are available upon request.

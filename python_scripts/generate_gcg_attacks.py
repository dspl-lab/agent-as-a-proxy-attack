"""
This script optimizes attack strings against extract-and-evaluate monitor using nanogcg engine.
Run as: python3 -m python_scripts.generate_gcg_attacks
"""

from python_scripts.gcg_engine import GCGEngine

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
gcg_engine = GCGEngine(
    attack_method="GCG",
    defense_method="eae",
    backend="nanogcg",
    model_id=model_id,
    num_steps=500,
    adv_string_init="x x x x x x x x x x x x x x x x x x x x",
    num_training_samples=1,
    search_width=512,
    use_prefix_cache=True,
    with_eos=False,
    target="<answer>0</answer>",
)

data_paths = [
    "results/aether-cot-monitoring/meta-llama/Llama-3.1-8B-Instruct/optim_str/slack/2025-12-27T03-24-58-08-00_run-hierarchical-monitor_WRNQKMECEccojce3E2b6oe.eval",
]

assert all([model_id in x for x in data_paths])

gcg_engine.load_data(data_paths, num_user_tasks=0)

gcg_engine.attack(save_suffix="slack", num_attacks=90)

# python3 -m python_scripts.generate_multigcg_attacks

from python_scripts.multigcg_engine import MultiGCGEngine

num_user_tasks = 1  # Only 1 is supported. We cannot do batch MultiGCG optimization.
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

gcg_engine = MultiGCGEngine(
    attack_method="MGCG_DT",
    defense_method="alignmentcheck",
    model_id=model_id,
    num_steps=500,
    adv_string_init="x x x x x x x x x x x x x x x x x x x x",
    num_training_samples=num_user_tasks,
    search_width=512,
    use_prefix_cache=True,
    with_eos=False,
)

gcg_engine.load_data(
    root_folders=[
        f"results/alignmentcheck/{model_id.replace("/", "_")}/alignmentcheck_results.jsonl"
    ],
    num_user_tasks=num_user_tasks,
)

gcg_engine.attack(save_suffix="threshold_05_test")

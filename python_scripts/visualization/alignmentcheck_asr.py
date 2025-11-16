# python3 -m python_scripts.visualization.alignmentcheck_asr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from python_scripts.utils import get_asr, get_existing_results, get_user_prompts
from itertools import zip_longest
import os
import random
import pandas as pd
import seaborn as sns

random.seed(1)

MODELS = [
    ("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.3"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"),
]

CATEGORIES = ["Baseline", "k=1", "k=4", "k=16"]

PATHS = {
    "base": "./results/alignmentcheck/{}/alignmentcheck_results.jsonl",
    "alignmentcheck": "./results/alignmentcheck/{}/universal_attack_results.jsonl",
}


def calc_ci(rates, confidence=0.95):
    """Calculates symmetric 95% CI error bars."""
    if not rates or len(rates) < 2:
        return 0.0
    n = len(rates)
    se = np.std(rates, ddof=1) / np.sqrt(n)
    return se * t.ppf((1 + confidence) / 2, df=n - 1)


def get_trace_filter():
    """Returns filter function for 'slack' and 'gpt-4o' traces."""
    prompts = get_user_prompts(
        base_path="./externals/agentdojo/runs", security_target=True
    )
    valid_paths = {"/".join(p["trace_path"].split("/")[2:]) for p in prompts}

    return lambda x: (
        (
            x["trace_path"] in valid_paths
            or "/".join(x["trace_path"].split("/")) in valid_paths
        )
        and "slack" in x["trace_path"]
        and "gpt-4o-2024-05-13" in x["trace_path"]
    )


def fetch_data(model_folder, category, row_filter, return_rows=False):
    """Extracts ASR rates based on category logic."""
    if category == "Baseline":
        path = PATHS["base"].format(model_folder)
        runs = get_existing_results([path], filter_=row_filter)
        return [get_asr(runs=runs)[0] * 100]

    path = (
        PATHS["pg+alignmentcheck"].format(model_folder)
        if "(PG)" in category
        else PATHS["alignmentcheck"].format(model_folder)
    )
    k_target = "1_mgcg" if "(PG)" in category else category.split("=")[1]

    runs = get_existing_results(
        [path], key=["optim_str", "trace_path"], filter_=row_filter
    )
    filtered_runs = [r for r in runs if str(r["num_training_samples"]) == k_target]

    optim_strs = set(r["optim_str"] for r in filtered_runs)
    if return_rows:
        rows = [[r for r in filtered_runs if r["optim_str"] == s] for s in optim_strs]
        if len(rows) <= 5:
            return rows

        return random.sample(rows, k=5)

    return [
        get_asr(runs=[r for r in filtered_runs if r["optim_str"] == s])[0] * 100
        for s in optim_strs
    ]


def main():
    row_filter = get_trace_filter()

    plot_data = {model[0]: {"ASR": [], "cis": []} for model in MODELS}
    plot_data = []
    for cat in CATEGORIES:
        for model_name, model_folder in MODELS:
            rates = fetch_data(model_folder, cat, row_filter)
            for rate in rates:
                plot_data.append(
                    {
                        "group": cat,
                        "model": model_name,
                        "ASR": rate,
                    }
                )
            if cat == "Baseline":
                continue
            results = fetch_data(model_folder, cat, row_filter, return_rows=True)
            success = [
                bool(get_asr(runs=[e for e in elements if e]))
                for elements in zip_longest(*results, fillvalue=False)
            ]
            asr_at_5 = sum(success) * 100 / len(success)

            model_name_at = model_name + "@" + str(max(5, len(results)))
            plot_data.append({"group": cat, "model": model_name_at, "ASR": asr_at_5})

    plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.2})
    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        x="group",
        y="ASR",
        hue="model",
        capsize=0.1,
        errorbar="ci",
        order=["Baseline", "k=1", "k=4", "k=16"],
    )

    plt.tight_layout()
    plt.ylabel("ASR", fontsize=16, fontweight="bold")
    plt.xlabel("Attack methods", fontsize=16, fontweight="bold")
    plt.ylim(0, 100)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    output_filename = "alignmentcheck_asr.png"
    plt.savefig(os.path.join("media", output_filename))
    plt.savefig(os.path.join("media", output_filename).replace(".png", ".pdf"))
    print(f"\nPlot saved to media/{output_filename}")


if __name__ == "__main__":
    main()

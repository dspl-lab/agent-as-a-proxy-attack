# python3 -m python_scripts.visualization.full_vs_replay


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from python_scripts.utils import (
    get_asr,
    get_baseline,
    get_existing_results,
    get_user_prompts,
)


def calculate_confidence_interval_for_rates(
    success_rates: list[list[float]], confidence_level: float = 0.95
) -> np.ndarray:
    """
    Calculates the confidence interval for a mean of success rates from multiple experiments.

    This is suitable for small sample sizes (e.g., 3-30 runs) and uses the t-distribution.

    Args:
        success_rates: A list of lists of float values, where each inner list is the success rate
                       from a single experimental run (e.g., [[0.85, 0.90, 0.88], [0.75, 0.80, 0.82]]).
        confidence_level: The desired confidence level for the interval. Defaults to 0.95 (95%).

    Returns:
        A NumPy array with a shape of 2xN, where N is the number of inner lists in success_rates.
        The first row contains the lower bounds and the second row contains the upper bounds.
    """
    lower_bounds = []
    upper_bounds = []

    for rates in success_rates:
        if len(rates) == 1:
            lower_bounds.append(rates[0])
            upper_bounds.append(rates[0])
            continue
        rates_array = np.array(rates)

        n = len(rates_array)
        mean_rate = np.mean(rates_array)
        std_dev = np.std(rates_array, ddof=1)

        degrees_of_freedom = n - 1
        alpha = 1 - confidence_level
        t_critical = t.ppf(1 - alpha / 2, df=degrees_of_freedom)

        margin_of_error = t_critical * (std_dev / np.sqrt(n))

        lower_bound = mean_rate - margin_of_error
        upper_bound = mean_rate + margin_of_error

        lower_bounds.append(mean_rate - lower_bound)
        upper_bounds.append(upper_bound - mean_rate)

    return np.array([lower_bounds, upper_bounds])


def plot_results(
    data,
    group_labels=None,
    bar_labels=None,
    title="",
    ylabel="ASR",
    xlabel="",
    colors=None,
    figsize=(8, 6),
    dpi=300,
):
    if group_labels is None:
        group_labels = ["Qwen_Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"]
    if bar_labels is None:
        bar_labels = ["Replay", "No replay"]
    if colors is None:
        colors = ["#2E86AB", "#A23B72"]

    data = np.array(data)
    group1 = data[0:2]
    group2 = data[2:4]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bar_width = 0.35
    x = np.arange(len(group_labels))

    bars1 = ax.bar(  # noqa: F841
        x - bar_width / 2,
        [group1[0], group2[0]],
        bar_width,
        label=bar_labels[0],
        color=colors[0],
        linewidth=1.2,
    )
    bars2 = ax.bar(  # noqa: F841
        x + bar_width / 2,
        [group1[1], group2[1]],
        bar_width,
        label=bar_labels[1],
        color=colors[1],
        linewidth=1.2,
    )

    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=16)

    ax.legend(frameon=False, fontsize=16, loc="best")

    # Customize grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    # Customize spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Tight layout
    plt.tight_layout()
    plt.savefig("./media/full_vs_replay.png", dpi=300, bbox_inches="tight")
    plt.savefig("./media/full_vs_replay.pdf", bbox_inches="tight")


DATA_SUBFOLDER_NAME_1 = "Qwen_Qwen2.5-7B-Instruct"
DATA_SUBFOLDER_NAME_2 = "mistralai_Mistral-7B-Instruct-v0.3"

groups = {
    "Qwen2.5-7B-Instruct (Replay)": f"./results/alignmentcheck/{DATA_SUBFOLDER_NAME_1}/universal_attack_results.jsonl",
    "Qwen2.5-7B-Instruct (Once)": f"./results/alignmentcheck/{DATA_SUBFOLDER_NAME_1}/universal_attack_results_scan_once.jsonl",
    "Mistral-7B-Instruct-v0.3 (Replay)": f"./results/alignmentcheck/{DATA_SUBFOLDER_NAME_2}/universal_attack_results.jsonl",
    "Mistral-7B-Instruct-v0.3 (Once)": f"./results/alignmentcheck/{DATA_SUBFOLDER_NAME_2}/universal_attack_results_scan_once.jsonl",
}

results = {}
user_prompts = get_user_prompts()
valid_trace_paths = [i["trace_path"] for i in user_prompts]


def filter_(x):
    return ("./externals/" + x["trace_path"]) in valid_trace_paths


# filter_ = lambda x: x["trace_path"].split("/")[2] in AGENTDOJO_MODELS
result_count = len(get_existing_results([next(iter(groups.values()))], filter_=filter_))
for group, path in groups.items():
    if "GCG" in group:
        group_results = get_existing_results(
            [path], key=["optim_str", "trace_path"], filter_=filter_
        )
    else:
        group_results = get_existing_results([path], filter_=filter_)
    # assert len(group_results) == result_count, f"Result counts don't match: {len(group_results)} vs. {result_count}"
    splits = []
    if group == "Base":
        results[group] = [get_baseline(runs=group_results)[0]]
    elif "(Transfered)" in group:
        for k in (1, 4, 16):
            group_name = group + f" (k={k})"
            results[group_name] = []
            runs = [
                i for i in group_results if str(i["num_training_samples"]) == str(k)
            ]
            optim_strings = list(set([i["optim_str"] for i in runs]))

            for string in optim_strings:
                temp_runs = [i for i in runs if i["optim_str"] == string]
                print(
                    group_name,
                    "Number of runs:",
                    len(temp_runs),
                    "ASR:",
                    get_asr(runs=temp_runs)[0],
                    "ER:",
                    get_asr(runs=temp_runs)[1],
                )
                results[group_name].append(get_asr(runs=temp_runs)[0])
    else:
        results[group] = [get_asr(runs=group_results)[0]]

print(results)
values_nonpg2 = []
conf_nonpg2 = []

for bar in groups.keys():
    v1 = np.array(results.get(bar, [0])) * 100
    values_nonpg2.append(np.mean(v1))
    conf_nonpg2.append(calculate_confidence_interval_for_rates([v1]))
conf_nonpg2 = np.concatenate(conf_nonpg2, axis=1)

print(values_nonpg2)
print(conf_nonpg2)
plot_results(values_nonpg2)

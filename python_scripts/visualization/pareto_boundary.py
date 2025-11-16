# python3 -m python_scripts.visualization.pareto_boundary
# TODO: Add false positive rate as a second line on the same graph

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from python_scripts.config import PARETO_MODELS
from python_scripts.utils import (
    get_asr,
    get_existing_results,
    get_f1_score,
    get_metadata,
    get_user_prompts,
    get_normalized_key,
)

# Set style for academic papers
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.size"] = 16
plt.rcParams["axes.linewidth"] = 1.2


def get_pareto_front(points: np.ndarray, maximize: List[bool]) -> np.ndarray:
    """
    Finds the indices of the non-dominated points (Pareto front).

    A point is non-dominated if no other point is better on all objectives
    and strictly better on at least one.

    Args:
        points: A numpy array of shape (n_points, n_objectives).
        maximize: A list/array of booleans indicating whether to maximize
                  (True) or minimize (False) each objective.

    Returns:
        A boolean array of shape (n_points,) indicating the Pareto-efficient
        points (True = on the front).
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)

    adjusted_points = points.copy()
    for i, max_obj in enumerate(maximize):
        if not max_obj:
            adjusted_points[:, i] = -adjusted_points[:, i]

    for i in range(adjusted_points.shape[0]):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                adjusted_points[is_efficient] > adjusted_points[i], axis=1
            ) | np.all(adjusted_points[is_efficient] == adjusted_points[i], axis=1)
            is_efficient[i] = True

            current_is_efficient = np.ones(points.shape[0], dtype=bool)

            for j in range(points.shape[0]):
                if i != j:
                    dominates_i = np.all(
                        adjusted_points[j] >= adjusted_points[i]
                    ) and np.any(adjusted_points[j] > adjusted_points[i])
                    if dominates_i:
                        current_is_efficient[i] = False
                        break
            is_efficient[i] = current_is_efficient[i]

    return is_efficient


def visualize_pareto_boundary(
    data: List[Tuple[str, float, float]],
    model_size_metric: str = "Model Size (Billions of Parameters)",
    asr_metric: str = "ASR",
    model_size_minimized: bool = True,
    asr_minimized: bool = True,
    filename: str = "pareto",
):
    """
    Visualizes the Pareto boundary for model size vs. ASR.

    Args:
        data: A list of tuples (model_name, model_size, ASR).
        model_size_metric: Label for the X-axis.
        asr_metric: Label for the Y-axis.
        model_size_minimized: True if smaller model size is better.
        asr_minimized: True if smaller ASR is better (more robust).
    """
    data = [(i[0], i[1], i[2] * 100) for i in data]
    names, sizes, asrs = zip(*data)
    points = np.array(list(zip(sizes, asrs)))

    if model_size_minimized and asr_minimized:
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i in range(points.shape[0]):
            if is_efficient[i]:
                dominates = np.all(points <= points[i], axis=1) & np.any(
                    points < points[i], axis=1
                )
                is_efficient[i] = not np.any(dominates)
    else:
        adjusted_points = points.copy()
        if model_size_minimized:
            adjusted_points[:, 0] = -adjusted_points[:, 0]
        if asr_minimized:
            adjusted_points[:, 1] = -adjusted_points[:, 1]

        is_efficient = np.ones(adjusted_points.shape[0], dtype=bool)
        for i in range(adjusted_points.shape[0]):
            if is_efficient[i]:
                dominates_i = np.all(
                    adjusted_points >= adjusted_points[i], axis=1
                ) & np.any(adjusted_points > adjusted_points[i], axis=1)
                is_efficient[i] = not np.any(dominates_i)

    pareto_points = points[is_efficient]
    pareto_names = np.array(names)[is_efficient]

    sort_index = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sort_index]
    pareto_names = pareto_names[sort_index]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        label="All Models",
        color="#808080",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        label="Pareto Front Models",
        color="#C00000",
        zorder=5,
        s=60,
        edgecolors="black",
        linewidth=0.8,
    )

    ax.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        color="#C00000",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Pareto Front",
    )

    for i, (name, x, y) in enumerate(zip(names, points[:, 0], points[:, 1])):
        ha = "left"
        xytext = (5, 5)
        if x > 6:
            ha = "right"
            xytext = (-5, 5)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=14,
            color="black" if not is_efficient[i] else "#8B0000",
            fontweight="normal" if not is_efficient[i] else "bold",
        )

    ax.set_xlabel(model_size_metric, fontsize=14, fontweight="bold")
    ax.set_ylabel(asr_metric, fontsize=14, fontweight="bold")

    ax.set_ylim(0, 100)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, loc="best", framealpha=0.9, edgecolor="black")

    plt.tight_layout()
    plt.savefig(f"./media/{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"./media/{filename}.pdf", bbox_inches="tight")

    print(f"Figure saved to ./media/{filename}.pdf")


# SECURITY=TRUE
insecure_model_data = []
secure_model_data = []
total_model_data = []
balanced_model_data = []
for model_name, num_params, _ in PARETO_MODELS:
    print(model_name, num_params)
    user_prompts = get_user_prompts(
        base_path="./externals/agentdojo/reruns/optim_str", security_target=None
    ) + get_user_prompts(base_path="./externals/agentdojo/runs", security_target=None)
    print(len(user_prompts))
    valid_trace_paths = [get_normalized_key(i["trace_path"]) for i in user_prompts]

    results_paths = [
        f"./results/alignmentcheck/{model_name.replace('/', '_')}/alignmentcheck_results_alignmentcheck.jsonl",
        f"./results/alignmentcheck/{model_name.replace('/', '_')}/alignmentcheck_results_base.jsonl",
        f"./results/alignmentcheck/{model_name.replace('/', '_')}/alignmentcheck_results.jsonl",
    ]
    results = get_existing_results(
        results_paths=results_paths,
        key=["model_name", "trace_path"],
        filter_=lambda x: get_normalized_key(x["trace_path"]) in valid_trace_paths
        and (
            "reruns/optim_str" in x["trace_path"] or "agentdojo/runs" in x["trace_path"]
        )
        and "gpt-4o-2024-05-13" in x["trace_path"],
    )
    if len(results) < 10:
        print("too short")
        print(len(results))
        print(results_paths)
        continue
    print(model_name, len(results))
    insecure_results = [i for i in results if get_metadata(i)["security"]]
    secure_results = [i for i in results if not get_metadata(i)["security"]]
    total_results = insecure_results + secure_results
    insecure_asr, insecure_error_rate = get_asr(
        runs=insecure_results, insecure_only=True
    )
    if insecure_asr and insecure_error_rate < 0.1:
        insecure_model_data.append((model_name.split("/")[1], num_params, insecure_asr))
    else:
        print(model_name.split("/")[1], num_params, insecure_asr)

    secure_asr, secure_error_rate = get_asr(runs=secure_results, insecure_only=False)
    if secure_asr and secure_error_rate < 0.1:
        secure_model_data.append((model_name.split("/")[1], num_params, secure_asr))

    total_asr, total_error_rate = get_asr(runs=total_results, insecure_only=False)
    if total_asr and total_error_rate < 0.1:
        total_model_data.append((model_name.split("/")[1], num_params, total_asr))

    if (
        insecure_error_rate < 0.1
        and secure_error_rate < 0.1
        and insecure_asr
        and secure_asr
    ):
        f1_score, error_rate = get_f1_score(runs=total_results)
        balanced_model_data.append((model_name.split("/")[1], num_params, f1_score))

print(insecure_model_data)
visualize_pareto_boundary(insecure_model_data, filename="pareto_insecure")

visualize_pareto_boundary(
    secure_model_data,
    asr_metric="Accuracy",
    asr_minimized=False,
    filename="pareto_secure",
)

visualize_pareto_boundary(total_model_data, filename="pareto")

visualize_pareto_boundary(
    balanced_model_data,
    asr_metric="F1 score",
    asr_minimized=False,
    filename="pareto_balanced",
)

# python3 -m python_scripts.tables.final
import glob
from python_scripts.utils import get_asr
from itertools import zip_longest
import os
import random
import pandas as pd
from python_scripts.visualization.alignmentcheck_asr import get_trace_filter, fetch_data
from python_scripts.visualization.eae_asr import get_asr as get_asr_eae

random.seed(1)

MODELS = [
    ("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.3"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"),
]

CATEGORIES = ["Baseline", "k=1", "k=4", "k=16"]

ALIGNMENTCHECK_PATHS = {
    "base": "./results/alignmentcheck/{}/alignmentcheck_results_alignmentcheck.jsonl",
    "alignmentcheck": "./results/alignmentcheck/{}/universal_attack_results.jsonl",
}

PG_ALIGNMENTCHECK_PATHS = {
    "base": "./results/alignmentcheck/{}/alignmentcheck_results_promptguard_toolcheck_only_alignmentcheck_normal.jsonl",
    "alignmentcheck": "./results/alignmentcheck/{}/universal_attack_results_promptguard_toolcheck_only_alignmentcheck_normal.jsonl",
}


def merge_boolean_lists(list_of_lists):
    return [any(elements) for elements in zip_longest(*list_of_lists, fillvalue=False)]


def get_alignmentcheck_df(paths):
    row_filter = get_trace_filter()

    plot_data = {model[0]: {"ASR@1": [], "cis": []} for model in MODELS}
    plot_data = []
    for cat in CATEGORIES:
        for model_name, model_folder in MODELS:
            rates = fetch_data(model_folder, cat, row_filter, paths=paths)
            for rate in rates:
                plot_data.append(
                    {
                        "Monitor": cat,
                        "Model": model_name,
                        "ASR@1": rate,
                    }
                )
            if cat == "Baseline":
                continue
            results = fetch_data(
                model_folder, cat, row_filter, return_rows=True, paths=paths
            )
            success = [
                bool(get_asr(runs=[e for e in elements if e]))
                for elements in zip_longest(*results, fillvalue=False)
            ]
            asr_at_5 = sum(success) * 100 / len(success)

            model_name_at = model_name + "@" + str(max(5, len(results)))
            plot_data.append(
                {"Monitor": cat, "Model": model_name_at, "ASR@1": asr_at_5}
            )
            for row in plot_data:
                if row["Monitor"] == "Baseline" and row["Model"] == model_name:
                    original_asr = row["ASR@1"]
                    plot_data.append(
                        {
                            "Monitor": "Baseline",
                            "Model": model_name_at,
                            "ASR@1": original_asr,
                        }
                    )
                    break
    return pd.DataFrame(data=plot_data)


def get_eae_df():
    suite = "slack"
    mode = "evaluator"  # e2e, extractor, evaluator
    k_values = [1, 4, 16]
    model_map = {
        "Mistral-7B": "mistralai/mistral-7b-instruct-v0.3",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen-72B": "qwen/qwen-2.5-72b-instruct",
    }

    attack_ids_dict = {
        1: {
            "Mistral-7B": ["0", "1", "2", "3", "4"],
            "Qwen-72B": ["4", "13", "36"],
            "Qwen2.5-7B": ["4", "13", "16", "36", "47"],
            # "Qwen2.5-7B": ["0", "4", "5", "10", "13", "16", "31", "36", "47"]
        },
        4: {
            "Mistral-7B": ["0", "4", "8", "12", "16"],
            "Qwen-72B": [],
            # "Qwen2.5-7B": ["0", "4", "8", "12", "16", "20", "24", "28", "32", "mistral_0", "mistral_4", "mistral_8", "mistral_12", "mistral_16"]
            "Qwen2.5-7B": ["4", "8", "12", "16", "28"],
        },
        16: {
            "Mistral-7B": ["0_resampled", "0", "16", "32", "48"],
            "Qwen-72B": [],
            "Qwen2.5-7B": ["0_resampled", "0", "16", "32", "48"],
        },
    }
    plot_data = []

    original_asr_for_baseline = {model_name: [] for model_name in model_map.keys()}

    for k in k_values:
        for model_name, model_id in model_map.items():
            attack_ids = attack_ids_dict[k][model_name]

            original_dataset_glob = f"results/aether-cot-monitoring/{model_id}/original/{suite}/*_run-hierarchical-monitor_*.eval"
            original_dataset_files = glob.glob(original_dataset_glob, recursive=True)
            if not original_dataset_files:
                original_dataset_glob = f"results/aether-cot-monitoring/{model_id}/original/{suite}/*_run-monitor_*.eval"
                original_dataset_files = glob.glob(
                    original_dataset_glob, recursive=True
                )

            if not original_dataset_files:
                continue

            original_dataset_path = max(original_dataset_files, key=os.path.getctime)

            original_success_by_id = []
            attack_success_by_id = []

            for id_ in attack_ids:
                base_path_glob = f"results/aether-cot-monitoring/{model_id}/optim_str/{mode}/{suite}/k_{k}/{id_}/*run-hierarchical-monitor*.eval"
                attack_dataset_files = glob.glob(base_path_glob, recursive=True)

                if not attack_dataset_files:
                    continue

                attack_dataset_path = max(attack_dataset_files, key=os.path.getctime)

                original_asr, eae_asr = get_asr_eae(
                    original_dataset_path, attack_dataset_path
                )
                original_samples, eae_samples = get_asr_eae(
                    original_dataset_path, attack_dataset_path, return_samples=True
                )

                original_success_by_id.append(original_samples)
                attack_success_by_id.append(eae_samples)

                plot_data.append(
                    {"Monitor": f"k={k}", "Model": model_name, "ASR@1": eae_asr}
                )
                original_asr_for_baseline[model_name].append(original_asr)

            if attack_success_by_id:
                model_name_at = model_name + "@" + str(len(attack_success_by_id))
                plot_data.append(
                    {
                        "Monitor": f"k={k}",
                        "Model": model_name_at,
                        "ASR@1": sum(merge_boolean_lists(attack_success_by_id))
                        / max(map(len, attack_success_by_id)),
                    }
                )
                plot_data.append(
                    {
                        "Monitor": "Baseline",
                        "Model": model_name_at,
                        "ASR@1": original_asr,
                    }
                )

    for model_name, asrs in original_asr_for_baseline.items():
        for asr in asrs:
            plot_data.append({"Monitor": "Baseline", "Model": model_name, "ASR@1": asr})

    df = pd.DataFrame(data=plot_data)
    df["ASR@1"] *= 100
    return df


def add_column(df):
    df["Model_clean"] = df["Model"].str.replace(r"@5$", "", regex=True)

    df1 = df[~df["Model"].str.endswith("@5")].copy()
    df5 = df[df["Model"].str.endswith("@5")].copy()
    df5 = df5.rename(columns={"ASR@1": "ASR@5"})

    out = df1.merge(
        df5[["Monitor", "Model_clean", "ASR@5"]],
        on=["Monitor", "Model_clean"],
        how="left",
    )
    out = out.rename(columns={"Model_clean": "Model"})
    out = out.loc[:, ~out.columns.duplicated()]

    return out


def collapse(df):
    baseline = df[df["Monitor"] == "Baseline"]
    collapsed = (
        df[df["Monitor"] != "Baseline"]
        .groupby(["Monitor", "Model"], as_index=False)["ASR@1"]
        .mean()
    )

    result = pd.concat([baseline, collapsed], ignore_index=True)

    result = result.sort_values(["Monitor", "Model"]).reset_index(drop=True)

    return add_column(result).drop_duplicates(["Monitor", "Model"])


def combine(names, dfs):
    for name, df in zip(names, dfs):
        df["source"] = name
    df = pd.concat(dfs, ignore_index=True)
    baseline_map = df[df["Monitor"] == "Baseline"].set_index(["source", "Model"])[
        "ASR@1"
    ]
    df["Baseline"] = df.apply(
        lambda row: baseline_map.loc[(row["source"], row["Model"])], axis=1
    )
    df = df[df["Monitor"] == "k=16"]
    df["Monitor"] = df["source"]
    df = df[["Monitor", "Model", "Baseline", "ASR@1", "ASR@5"]]

    return df


def df_to_latex_table(df):
    lines = []
    lines.append(r"\begin{tabular}{l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Monitor & Model & Baseline & ASR@1 & ASR@5 \\")
    lines.append(r"\midrule")

    for monitor, group in df.groupby("Monitor", sort=False):
        rows = group.sort_values("Model")
        n = len(rows)

        for i, (_, row) in enumerate(rows.iterrows()):
            if i == 0:
                lines.append(
                    r"\multirow{{{}}}{{*}}{{{}}} & {} & {} & {} & {} \\".format(
                        n,
                        monitor,
                        row["Model"],
                        row["Baseline"],
                        row["ASR@1"],
                        row["ASR@5"],
                    )
                )
            else:
                lines.append(
                    r" & {} & {} & {} & {} \\".format(
                        row["Model"], row["Baseline"], row["ASR@1"], row["ASR@5"]
                    )
                )

        lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def main():
    alignmentcheck_df = collapse(get_alignmentcheck_df(ALIGNMENTCHECK_PATHS))
    pg_alignmentcheck_df = collapse(get_alignmentcheck_df(PG_ALIGNMENTCHECK_PATHS))
    eae_df = collapse(get_eae_df())

    names = ["AlignmentCheck", "PromptGuard 2 + AlignmentCheck", "Extract-and-evaluate"]
    final = combine(names, [alignmentcheck_df, pg_alignmentcheck_df, eae_df])
    final = final.round(2)
    order = ["AlignmentCheck", "PromptGuard 2 + AlignmentCheck", "Extract-and-evaluate"]
    final = final.set_index("Monitor").loc[order].reset_index()
    print(df_to_latex_table(final))


if __name__ == "__main__":
    main()

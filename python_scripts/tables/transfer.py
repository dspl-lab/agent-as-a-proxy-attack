# python3 -m python_scripts.tables.transfer2

import glob
import json
import os
from collections import defaultdict

import pandas as pd

# Added norm_id to imports as requested
from python_scripts.utils import get_asr, get_existing_results, get_metadata


def get_normalized_key(sample_id: str):
    """Normalizes ID to handle different split depths in file paths."""
    parts = sample_id.split("/")
    return tuple(parts[-5:])


agent = "gpt-4o-2024-05-13"
min_accepted_result_count = 50

source_models = (
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
)

target_models = (
    "meta-llama/llama-3.1-405b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "openai/gpt-oss-20b",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen-2.5-72B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B-FP8",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
)
model_params_map = {
    # Keys are the short model names (e.g., "8B-Instruct" from "Llama-3.1-8B-Instruct")
    "3B-Instruct": 3,
    "4B-Instruct-2507": 4,
    "8B-FP8": 8,
    "8B-Instruct": 8,  # For Llama-3.1-8B-Instruct
    "7b-v1.5": 7,  # For vicuna-7b-v1.5
    "7B-Instruct": 7,  # For Qwen2.5-7B-Instruct
    "Mistral-7B-Instruct-v0.3": 7,  # For mistral-7b-instruct-v0.3
    "gpt-4o-2024-05-13": 175,  # Placeholder, actual value unknown
    "gpt-4o-mini-2024-07-18": 13,  # Placeholder, actual value unknown
}


def load_file_data(path):
    """
    Reads a JSONL file and returns a list of data objects
    and a set of normalized trace_ids found in that file.
    """
    data_list = []
    ids_set = set()

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    trace_path = data.get("trace_path")
                    if trace_path:
                        nid = get_normalized_key(trace_path)
                        ids_set.add(nid)
                        data_list.append(data)
                except json.JSONDecodeError:
                    continue
    return data_list, ids_set


def get_summary_df():
    data_dir = "./results/alignmentcheck"

    all_source_models = set()
    all_target_models = set()

    # Store paths to process later
    transfer_file_paths = []

    # Discover all source-target pairs
    for target_model_sanitized in target_models:
        target_dir = os.path.join(data_dir, target_model_sanitized.replace("/", "_"))
        print(target_dir)
        for result_file in glob.glob(
            os.path.join(target_dir, "transfer_from_*_results.jsonl")
        ):
            transfer_file_paths.append(result_file)
            # Extract source model from filename
            source_model_sanitized = result_file.split("transfer_from_")[1].replace(
                "_results.jsonl", ""
            )
            all_source_models.add(source_model_sanitized.replace("_", "/"))
            all_target_models.add(target_model_sanitized.replace("_", "/"))

    # --- Phase 1: Load Data and Determine Common Intersection ---

    # Cache for loaded data: path -> list of dicts
    file_data_cache = {}

    # The intersection of IDs across ALL files (transfer and base)
    common_ids = None

    # 1.1 Load Transfer Files
    for path in transfer_file_paths:

        def filter_(x):
            return agent in x["trace_path"] and "reruns/optim_str" in x["trace_path"]

        print(path)
        print("-------")

        data_list = get_existing_results(
            results_paths=[path],
            key=["num_training_samples", "trace_path"],
            filter_=filter_,
        )

        ids_set = set([get_normalized_key(i["trace_path"]) for i in data_list])
        print(len(ids_set))
        print(list(ids_set)[0])

        if not data_list:
            continue  # Skip empty files

        file_data_cache[path] = data_list

        if common_ids is None:
            common_ids = ids_set
        else:
            print("len:", len(common_ids))
            common_ids.intersection_update(ids_set)

    # 1.2 Load Base Files (for all targets)
    # We map target_model -> base_file_path to process them similarly
    base_file_paths = {}

    for target_model in all_target_models:
        target_model_sanitized = target_model.replace("/", "_")
        base_path = f"./experiments/data/{target_model_sanitized}/alignmentcheck_results_base.jsonl"

        # Only check existence here; detailed load happens in loop
        if os.path.exists(base_path):
            base_file_paths[target_model] = base_path

            def filter_(x):
                return (
                    agent in x["trace_path"]
                    and "reruns/optim_str" in x["trace_path"]
                    and get_metadata(x)["security"]
                )

            data_list = get_existing_results(
                results_paths=[base_path],
                key=["num_training_samples", "trace_path"],
                filter_=filter_,
            )
            ids_set = set([get_normalized_key(i["trace_path"]) for i in data_list])

            print(len(ids_set))
            print(list(ids_set)[0])

            if not data_list:
                continue

            file_data_cache[base_path] = data_list

            if common_ids is None:
                common_ids = ids_set
            else:
                print("len:", len(common_ids))
                common_ids.intersection_update(ids_set)

    if common_ids is None:
        common_ids = set()

    print(f"Number of common samples across all experiments: {len(common_ids)}")

    if len(common_ids) == 0:
        print("Warning: No common samples found. Returning empty DataFrame.")
        return pd.DataFrame()

    # --- Phase 2: Process Data using only Common IDs ---

    results = defaultdict(lambda: defaultdict(dict))

    # 2.1 Process Transfer Results
    for path in transfer_file_paths:
        if path not in file_data_cache:
            continue

        target_model_sanitized = os.path.basename(os.path.dirname(path))
        source_model_sanitized = (
            os.path.basename(path)
            .split("transfer_from_")[1]
            .replace("_results.jsonl", "")
        )

        source_model = source_model_sanitized.replace("_", "/")
        target_model = target_model_sanitized.replace("_", "/")

        # Filter raw data based on common_ids
        raw_data = file_data_cache[path]
        filtered_data = [
            d for d in raw_data if get_normalized_key(d.get("trace_path")) in common_ids
        ]

        # print(len(filtered_data))

        # Group by num_training_samples
        grouped_results = defaultdict(list)
        for data in filtered_data:
            k = data.get("num_training_samples")
            if k is not None:
                grouped_results[k].append(data)

        for k, k_results in grouped_results.items():
            if len(k_results) < min_accepted_result_count:
                continue
            asr = get_asr(runs=k_results)[0]
            results[(source_model.split("/")[1], target_model.split("/")[1])][
                f"k={k}"
            ] = asr

    # 2.2 Process Base Results
    base_asr_map = {}

    for target_model, path in base_file_paths.items():
        if path not in file_data_cache:
            continue

        raw_data = file_data_cache[path]
        filtered_data = [
            d for d in raw_data if get_normalized_key(d.get("trace_path")) in common_ids
        ]

        if len(filtered_data) < min_accepted_result_count:
            # print(f"Base results for {target_model} insufficient after filtering: {len(filtered_data)}")
            continue

        base_asr = get_asr(runs=filtered_data)[0]
        base_asr_map[target_model.split("/")[-1]] = base_asr

    # --- Phase 3: Build DataFrame ---

    # Create a multi-index dataframe
    df_data = []
    for (source_model, target_model), k_results in results.items():
        row = {"source model": source_model, "target model": target_model}
        row.update(k_results)
        df_data.append(row)

    if not df_data:
        return pd.DataFrame()

    df = pd.DataFrame(df_data)

    # Add a temporary column for sorting by target model parameters
    df["target_model_params"] = df["target model"].map(
        lambda x: model_params_map.get(x, 0)
    )

    # Sort by target model parameters before setting index
    df = df.sort_values(by="target_model_params")

    # Set index and drop the temporary sorting column
    df = df.set_index(["source model", "target model"]).drop(
        columns="target_model_params"
    )
    df.columns = [c.replace("k=", "k") for c in df.columns]

    df["base"] = df.index.get_level_values("target model").map(base_asr_map)

    # Define the desired columns for the ASR supercolumn
    desired_cols_order = ["base", "k1", "k4", "k16"]

    # Filter for desired columns that are present in the DataFrame
    final_cols = [col for col in desired_cols_order if col in df.columns]
    df = df[final_cols]

    # Create the multi-level column structure
    multi_index_tuples = []
    for col in final_cols:
        if col == "base":
            multi_index_tuples.append(("ASR", "base"))
        else:
            # Assumes column format is 'k' followed by a number, e.g., 'k1', 'k4'
            k_val = col[1:]
            multi_index_tuples.append(("ASR", f"k={k_val}"))

    df.columns = pd.MultiIndex.from_tuples(multi_index_tuples)

    df = df.sort_index()
    df = df.fillna(0)

    return df


def main():
    df = get_summary_df()

    if df.empty:
        print("No data found.")
        return

    # Reset index to convert to regular columns
    df = df.reset_index()

    # Create multi-index columns for ALL columns
    new_columns = [("", "source model"), ("", "target model")] + [
        ("ASR", col[1]) for col in df.columns[2:]
    ]
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    print(
        df.to_latex(
            float_format="%.3f",
            column_format="llcccc",
            multicolumn=True,
            multicolumn_format="c",
            index=False,
        )
    )


if __name__ == "__main__":
    main()

# python3 -m python_scripts.generate_attacks

import json
import sys
import glob
import torch
import os
import random
import nanogcg
from dataclasses import asdict
from tqdm import tqdm
from inspect_ai.log import read_eval_log
from python_scripts.utils import (
    load_model,
    get_user_prompts,
    get_targets,
    get_trace,
    get_existing_results,
)

from llamafirewall import (
    ToolMessage,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "externals/AdaptiveAttackAgent"))
from attacks import get_method_class  # noqa: E402

seed = 15
random.seed(seed)


def normalize_id(target_id):
    if target_id.startswith("./externals"):
        return "/".join(target_id.split("/")[2:])

    if target_id.startswith("externals"):
        return "/".join(target_id.split("/")[1:])

    return target_id


class GCGEngine:
    def __init__(
        self,
        backend="nanogcg",  # nanogcg or AdaptiveAttackAgent
        attack_method="GCG",
        defense_method="alignmentcheck",  # or "eae" (extract-and-evaluate)
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        num_steps=500,
        adv_string_init=" ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        search_width=512,
        use_prefix_cache=True,
        with_eos=False,
        target="<answer>0</answer>",
        num_training_samples=1,
        not_allowed_tokens=None,
    ):
        self.backend = backend
        self.attack_method_str = attack_method
        self.attack_method = get_method_class(attack_method)
        self.defense_method = defense_method
        self.model_id = model_id
        self.with_eos = with_eos
        self.target = target
        self.num_training_samples = num_training_samples
        self.gcg_config = {
            "num_steps": num_steps,
            "adv_string_init": adv_string_init,
            "search_width": search_width,
            "use_prefix_cache": use_prefix_cache,
        }
        self.data = []
        self.not_allowed_tokens = not_allowed_tokens

    @staticmethod
    def _extract_input(sample):
        prefix = (
            "Below, you will be shown a user query that an AI agent "
            "was instructed to solve"
        )

        attachments = getattr(sample, "attachments", None)
        if not attachments:
            return None

        for _, text in attachments.items():
            if text.startswith(prefix):
                return text

        return None

    def load_data(
        self,
        root_folders: list,
        num_user_tasks: int,
        agentdojo_folder: str,
        agentdojo_llms: list,
        optim_str_count: int | None = None,
    ):
        case_id = 0
        if self.defense_method == "eae":
            eval_files = []
            for path in root_folders:
                if os.path.isdir(path):
                    pattern = os.path.join(path, "**", "*.eval")
                    found_files = glob.glob(pattern, recursive=True)
                    eval_files.extend(found_files)
                    print(f"Found {len(found_files)} .eval files in {path}")
                elif path.endswith(".eval"):
                    eval_files.append(path)
                else:
                    print(f"Warning: {path} is not a directory")
            for eval_file in eval_files:
                monitor_log = read_eval_log(eval_file)
                for sample in monitor_log.samples:
                    input_str = self._extract_input(sample)
                    splits = input_str.split("{optim_str}", 1)
                    if self.attack_method_str == "PGCG":
                        data_dict = {
                            "CaseID": case_id,
                            "Target": self.target,
                            "before_adv_string": splits[0],
                            "after_adv_string": splits[1],
                            "template": input_str,
                        }
                        pass
                    else:
                        if len(splits) != 2:
                            continue
                        data_dict = {
                            "CaseID": case_id,
                            "Target": self.target,
                            "before_adv_string": splits[0],
                            "after_adv_string": splits[1],
                        }
                    self.data.append(data_dict)
                    case_id += 1

        elif self.defense_method == "alignmentcheck":
            case_id = 0

            all_traces = get_user_prompts(
                base_path=agentdojo_folder,
                list_of_models=agentdojo_llms,
                return_string=True,
                include_optim_str=True,
                # max_length=13000
            )
            all_targets = get_targets("Qwen/Qwen2.5-7B-Instruct")
            all_targets = {
                normalize_id(i["trace_path"]): json.dumps(i["scan_result"])
                for i in all_targets
            }
            results_path = f"./results/alignmentcheck/{self.model_id.replace('/', '_')}/alignmentcheck_results_alignmentcheck.jsonl"

            def filter_(x):
                return x["decision"] == "ScanDecision.ALLOW"

            existing_results = get_existing_results(
                results_paths=[results_path], filter_=filter_
            )
            existing_results_paths = [i["trace_path"] for i in existing_results]
            for trace in all_traces:
                if not trace["messages"]:
                    continue
                target_id = normalize_id(trace["trace_path"])
                if (
                    trace["trace_path"] in existing_results_paths
                    or target_id in existing_results_paths
                ):
                    print(trace["trace_path"])
                    continue
                if optim_str_count and optim_str_count > trace["messages"].count(
                    "optim_str"
                ):
                    continue
                if self.attack_method_str == "PGCG":
                    data_dict = {
                        "CaseID": case_id,
                        "Target": all_targets[target_id],
                        "template": trace["messages"],
                    }
                else:
                    alignmentcheck_splits = trace["messages"].split("{optim_str}", 1)
                    if len(alignmentcheck_splits) != 2:
                        continue

                    with open(trace["trace_path"]) as f:
                        messages = get_trace(
                            json.load(f),
                            True,
                            "{optim_str}",
                            False,
                            drop_tool_trace=False,
                        )

                    for m in messages:
                        if isinstance(m, ToolMessage):
                            promptguard_splits = m.content.split("{optim_str}", 1)
                            if len(promptguard_splits) == 2:
                                break
                    if len(promptguard_splits) != 2:
                        continue

                    if not all_targets.get(target_id):
                        continue
                    data_dict = {
                        "CaseID": case_id,
                        "Target": all_targets[target_id],
                        "before_adv_string": alignmentcheck_splits[0],
                        "after_adv_string": alignmentcheck_splits[1],
                        "tool_output_before": promptguard_splits[0],
                        "tool_output_after": promptguard_splits[1],
                    }
                self.data.append(data_dict)
                case_id += 1

    def _nanogcg(self, model, tokenizer, messages, target):
        config = nanogcg.GCGConfig(
            topk=256,
            seed=42,
            buffer_size=1,
            optim_str_init=self.gcg_config["adv_string_init"],
            verbosity="DEBUG",
            use_prefix_cache=True,
            batch_size=32,
            num_steps=self.gcg_config["num_steps"],
            search_width=self.gcg_config["search_width"],
            not_allowed_ids=self.not_allowed_tokens,
        )

        return nanogcg.run(
            model,
            tokenizer,
            messages,
            [target for _ in range(self.num_training_samples)],
            config,
        )

    def attack(self, save_suffix=None, num_attacks: int = 1):
        print("Training adv string")
        model, tokenizer = load_model(self.model_id)

        result_dir = f"./results/attacks/{self.attack_method_str}/{self.defense_method}/{self.model_id}"
        if save_suffix:
            result_dir = result_dir + "/" + save_suffix

        shuffled_data = self.data.copy()
        random.shuffle(self.data)
        shuffled_data = shuffled_data[: num_attacks * self.num_training_samples]
        for i in tqdm(range(0, len(shuffled_data), self.num_training_samples)):
            training_samples = shuffled_data[i : i + self.num_training_samples]
            result_file = f"{result_dir}/{self.backend}/k_{self.num_training_samples}/{i}_seed_{seed}.json"
            os.makedirs(os.path.dirname(result_file), exist_ok=True)

            if os.path.exists(result_file):
                print(f"Skipping ID: {i}, Seed: {seed}")
                continue
            if self.backend == "nanogcg":
                print(f"Starting. ID: {i}, Seed: {seed}")
                messages = [
                    sample["before_adv_string"]
                    + "{optim_str}"
                    + sample["after_adv_string"]
                    for sample in training_samples
                ]
                try:
                    result = self._nanogcg(model, tokenizer, messages, self.target)
                except torch.OutOfMemoryError:
                    continue
                result = asdict(result)
                result["files"] = [sample["CaseID"] for sample in training_samples]
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=4)
            elif self.backend == "AdaptiveAttackAgent":
                adaptive_attack_method = self.attack_method(
                    model,
                    tokenizer,
                    not_allowed_tokens=self.not_allowed_tokens,
                    **self.gcg_config,
                )
                adaptive_attack_method.train_adv_string(
                    shuffled_data[i], result_file, with_eos=self.with_eos
                )

        return result_dir

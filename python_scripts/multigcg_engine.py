import json
import sys
import os
import random
from tqdm import tqdm
from python_scripts.utils import get_targets, get_user_prompts, get_trace, load_model

from llamafirewall import (
    ToolMessage,
)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "externals/AdaptiveAttackAgent"))
from attacks import get_method_class  # noqa: E402

random.seed(1)


class MultiGCGEngine:
    def __init__(
        self,
        backend="nanogcg",  # nanogcg or AdaptiveAttackAgent
        attack_method="MGCG_DT",
        defense_method="alignmentcheck",  # or "eae" (extract-and-evaluate)
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        num_steps=500,
        adv_string_init=" ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        search_width=512,
        use_prefix_cache=True,
        with_eos=False,
        num_training_samples=1,
    ):
        self.backend = backend
        self.attack_method_str = attack_method
        self.attack_method = get_method_class(attack_method)
        self.defense_method = defense_method
        self.model_id = model_id
        self.with_eos = with_eos
        self.num_training_samples = num_training_samples
        self.gcg_config = {
            "num_steps": num_steps,
            "adv_string_init": adv_string_init,
            "search_width": search_width,
            "use_prefix_cache": use_prefix_cache,
        }
        self.data = []

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

    def load_data(self, root_folders: list, num_user_tasks: int):
        case_id = 0

        all_traces = get_user_prompts(
            return_string=True,
            include_optim_str=True,
        )
        all_targets = get_targets("Qwen/Qwen2.5-7B-Instruct")
        all_targets = {
            i["trace_path"]: json.dumps(i["scan_result"]) for i in all_targets
        }

        for trace in all_traces:
            if not trace["messages"]:
                continue
            alignmentcheck_splits = trace["messages"].split("{optim_str}", 1)
            if len(alignmentcheck_splits) != 2:
                continue
            target_id = "/".join(trace["trace_path"].split("/")[2:])

            with open(trace["trace_path"]) as f:
                messages = get_trace(json.load(f), True, "{optim_str}", False)

            for m in messages:
                if isinstance(m, ToolMessage):
                    promptguard_splits = m.content.split("{optim_str}", 1)
                    if len(promptguard_splits) == 2:
                        break
            if len(promptguard_splits) != 2:
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

        case_id += 1

    def attack(self, save_suffix=None):
        print("Training adv string")
        model, tokenizer = load_model(self.model_id)
        adaptive_attack_method = self.attack_method(model, tokenizer, **self.gcg_config)

        result_dir = f"./results/attacks/{self.attack_method_str}/{self.model_id}"
        if save_suffix:
            result_dir = result_dir + "/" + save_suffix

        os.makedirs(result_dir, exist_ok=True)

        shuffled_data = self.data.copy()
        random.shuffle(self.data)
        for i in tqdm(range(0, len(self.data), self.num_training_samples)):
            result_file = f"{result_dir}/{shuffled_data[i]['CaseID']}.json"
            if "MGCG" in self.attack_method_str:
                adaptive_attack_method.train_adv_string(
                    shuffled_data[i], result_file, with_eos=self.with_eos
                )
            else:
                pass
        return result_dir

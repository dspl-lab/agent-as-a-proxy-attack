# python3 -m python_scripts.generate_targets

import json
import os

import outlines
import torch
from outlines.inputs import Chat
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from python_scripts.prompts import TARGET_GENERATOR_SYSTEM_PROMPT
from python_scripts.utils import get_targets, get_user_prompts

ALIGNMENTCHECK_LLM = "Qwen/Qwen2.5-7B-Instruct"
AGENTDOJO_MODELS = [
    "gpt-4o-2024-05-13",
    # "google_gemini-2.5-pro"
]


class ScanResult(BaseModel):
    observation: str
    thought: str
    conclusion: bool


def generate_target(user_prompt, model):
    messages = Chat(
        [
            {"role": "system", "content": TARGET_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt.split("# Input")[1].strip()},
        ]
    )

    scan_result = model(
        messages,
        ScanResult,
        max_new_tokens=500,
    )

    scan_result = ScanResult.model_validate_json(scan_result).model_dump()
    return scan_result


def main():
    all_traces = get_user_prompts(
        base_path="externals/agentdojo/reruns/optim_str",
        include_optim_str=False,
        return_string=True,
        list_of_models=AGENTDOJO_MODELS,
    )

    all_targets = get_targets()

    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(ALIGNMENTCHECK_LLM, device_map="auto"),
        AutoTokenizer.from_pretrained(ALIGNMENTCHECK_LLM),
    )

    results_path = f"./results/alignmentcheck/{ALIGNMENTCHECK_LLM.replace('/', '_')}/generated_targets.jsonl"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    for idx, trace in tqdm(enumerate(all_traces), total=len(all_traces)):
        try:
            if trace["trace_path"] in [i["trace_path"] for i in all_targets]:
                continue
            if not trace["messages"]:
                continue

            result = generate_target(trace["messages"], model)
            result = {
                "id": idx,
                "trace_path": trace["trace_path"],
                "scan_result": result,
            }
            with open(results_path, "a") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        except torch.OutOfMemoryError:
            continue


if __name__ == "__main__":
    main()

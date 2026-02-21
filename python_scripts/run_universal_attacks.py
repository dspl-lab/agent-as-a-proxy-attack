# CUDA_VISIBLE_DEVICES=2 python3 -m python_scripts.run_universal_attacks

import argparse
import os
import json

from tqdm import tqdm

from python_scripts.config import ALIGNMENTCHECK_LLM
from python_scripts.utils import (
    get_existing_results,
    get_user_prompts,
    log_scan_result,
    scanner_dict,
    load_model,
)
from llamafirewall import (
    LlamaFirewall,
)


def main(
    optim_str,
    num_training_samples,
    mode,
    alignmentcheck_llm,
    agentdojo_path,
    agentdojo_llms,
    engine,
    drop_tool_trace,
):
    model_id = alignmentcheck_llm.replace("/", "_")
    suffix = "_no_tool" if drop_tool_trace else ""
    mode_part = f"_{mode}" if mode != "alignmentcheck" else ""
    results_path = (
        f"./results/alignmentcheck/{model_id}/"
        f"universal_attack_results{mode_part}{suffix}.jsonl"
    )
    print(results_path)
    scanner_config = scanner_dict[mode]

    existing_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                existing_results[str(data["optim_str"]) + data["trace_path"]] = data

    if agentdojo_llms:
        all_traces = get_user_prompts(
            base_path=agentdojo_path,
            return_string=False,
            include_optim_str=True,
            optim_str=optim_str,
            security_target=True,
            max_length=16000,
            list_of_models=agentdojo_llms,
            drop_tool_trace=drop_tool_trace,
        )
    else:
        all_traces = get_user_prompts(
            base_path=agentdojo_path,
            return_string=False,
            include_optim_str=True,
            optim_str=optim_str,
            security_target=True,
            max_length=16000,
            drop_tool_trace=drop_tool_trace,
        )
    print("Total length:", len(all_traces))

    if args.engine == "huggingface":
        model, tokenizer = load_model(alignmentcheck_llm)

    if args.engine == "huggingface":
        firewall = LlamaFirewall(
            model=model,
            tokenizer=tokenizer,
            backend=engine,
            scanners=scanner_config,
        )
    else:
        firewall = LlamaFirewall(
            model_name=args.alignmentcheck_llm,
            backend=engine,
            scanners=scanner_config,
        )

    for idx, trace in tqdm(enumerate(all_traces), total=len(all_traces)):
        id = optim_str + trace["trace_path"]
        if existing_results.get(id, False):
            continue
        if mode == "scan_once":
            past_trace = [
                msg for ix, msg in enumerate(trace["messages"]) if ix < len(trace) - 1
            ]
            result = firewall.scan(
                trace["messages"][-1], past_trace if past_trace else None
            )
        else:
            result = firewall.scan_replay(trace["messages"])
        log_scan_result(
            result,
            trace["trace_path"],
            optim_str,
            num_training_samples,
            alignmentcheck_llm,
            idx,
            results_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="base")
    parser.add_argument("--num_training_samples", type=str, default="base")
    parser.add_argument("--alignmentcheck_llm", type=str, default=ALIGNMENTCHECK_LLM)
    parser.add_argument(
        "--agentdojo_path", type=str, default="./externals/agentdojo/runs"
    )
    parser.add_argument("--engine", type=str, default="huggingface")
    parser.add_argument("--agentdojo_llms", nargs="+")
    parser.add_argument("--drop_tool_trace", action="store_true")
    args = parser.parse_args()

    attacks = get_existing_results(
        [
            f"./results/alignmentcheck/{args.alignmentcheck_llm.replace('/', '_')}/universal_attacks.jsonl"
        ]
    )
    attacks = [i for i in attacks if i["id"] == str(args.num_training_samples)]
    print("Number of attacks:", len(attacks))
    for attack in attacks:
        main(
            attack["attack_result"]["best_string"],
            args.num_training_samples,
            args.mode,
            args.alignmentcheck_llm,
            args.agentdojo_path,
            args.agentdojo_llms,
            args.engine,
            args.drop_tool_trace,
        )

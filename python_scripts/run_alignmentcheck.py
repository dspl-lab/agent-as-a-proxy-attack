# CUDA_VISIBLE_DEVICES=0 python3 -m python_scripts.run_alignmentcheck

import argparse
import json
import os
from typing import Optional

from accelerate import Accelerator, PartialState
from tqdm import tqdm

from python_scripts.config import ALIGNMENTCHECK_LLM
from python_scripts.utils import get_user_prompts, scanner_dict, load_model
from llamafirewall import (
    LlamaFirewall,
    ScanResult,
    Trace,
)

accelerator = Accelerator()


def run_alignment_check(trace: Trace, firewall, no_replay) -> Optional[ScanResult]:
    """
    Run AlignmentCheck scanner on the provided trace.

    Args:
        trace: The conversation trace to analyze.

    Returns:
        Optional[ScanResult]: The scan result or None if an error occurred.
    """
    try:
        if no_replay:
            return firewall.scan_final(trace), None
        else:
            return firewall.scan_replay(trace), None
    except Exception as e:
        print("ERROR:", e)
        return None, e


def create_log_entry(
    result: Optional[ScanResult], e, run, aligmentcheck_llm, idx
) -> None:
    log = {}
    log["benchmark"] = "agentdojo"
    log["trace_index"] = idx
    log["model_name"] = aligmentcheck_llm
    log["trace_path"] = run["trace_path"]

    if result:
        log["score"] = (f"{result.score:.4f}",)
        log["decision"] = str(result.decision)
        log["reason"] = str(result.reason)
        log["status"] = str(result.status)
        log["alignmentcheck_error"] = None
    else:
        log["alignmentcheck_error"] = str(e)

    return log


def main() -> int:
    """
    Main function to run the demo.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentdojo_path", default="./externals/agentdojo/runs")
    parser.add_argument("--agentdojo_llms", nargs="+")
    parser.add_argument("--alignmentcheck_llm", default=ALIGNMENTCHECK_LLM)
    parser.add_argument("--engine", default="huggingface")
    parser.add_argument("--quantization", default="fp8")
    parser.add_argument("--mode", default="base")
    parser.add_argument("--drop_tool_trace", action="store_true")
    parser.add_argument("--no_replay", action="store_true")
    parser.add_argument(
        "--security",
        type=str,
        choices=["true", "false", "none"],
        default="none",
        help='Security level: "true", "false", or "none".',
    )
    args = parser.parse_args()
    if args.security == "true":
        args.security = True
    elif args.security == "false":
        args.security = False
    else:
        args.security = None

    with accelerator.main_process_first():
        scanner_config = scanner_dict[args.mode]

        no_replay_suffix = "_no_replay" if args.no_replay else ""
        no_tool_suffix = "_no_tool" if args.drop_tool_trace else ""

        results_path = (
            f"results/alignmentcheck/{args.alignmentcheck_llm.replace('/', '_')}/"
            f"alignmentcheck_results_{args.mode}{no_replay_suffix}{no_tool_suffix}.jsonl"
        )
        if args.engine == "huggingface":
            model, tokenizer = load_model(args.alignmentcheck_llm)

        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        existing_results = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    if data.get("status") == "ScanStatus.SUCCESS":
                        existing_results[data["trace_path"]] = data
    if args.engine == "huggingface":
        model = accelerator.prepare(model)

        firewall = LlamaFirewall(
            model=model,
            tokenizer=tokenizer,
            backend=args.engine,
            quantization=args.quantization,
            scanners=scanner_config,
            accelerator=accelerator,
        )
    else:
        firewall = LlamaFirewall(
            model_name=args.alignmentcheck_llm,
            backend=args.engine,
            quantization=args.quantization,
            scanners=scanner_config,
            accelerator=accelerator,
        )
    print("SECURITY:", args.security)
    if args.agentdojo_llms:
        runs = get_user_prompts(
            base_path=args.agentdojo_path,
            list_of_models=args.agentdojo_llms,
            include_optim_str=False,
            return_string=False,
            security_target=args.security,
            drop_tool_trace=args.drop_tool_trace,
        )
    else:
        runs = get_user_prompts(
            base_path=args.agentdojo_path,
            include_optim_str=False,
            return_string=False,
            security_target=args.security,
            drop_tool_trace=args.drop_tool_trace,
        )

    print(len(runs))
    progress_bar = tqdm(total=len(runs), disable=not accelerator.is_main_process)

    all_results = []
    with PartialState().split_between_processes(runs, apply_padding=True) as new_runs:
        for idx, run in enumerate(new_runs):
            if (
                existing_results.get(run["trace_path"], False)
                and existing_results.get(run["trace_path"], False)["status"]
                == "ScanStatus.SUCCESS"
            ):
                if accelerator.is_main_process:
                    progress_bar.update(accelerator.num_processes)
                continue
            result, e = run_alignment_check(run["messages"], firewall, args.no_replay)
            log_entry = create_log_entry(result, e, run, args.alignmentcheck_llm, idx)
            with open(results_path, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            all_results.append(log_entry)
            if accelerator.is_main_process:
                progress_bar.update(accelerator.num_processes)
    return 0


if __name__ == "__main__":
    main()

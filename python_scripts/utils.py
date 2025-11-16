import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
import itertools
import json
import os
import random

from python_scripts.config import (
    AGENTDOJO_ATTACKS,
    AGENTDOJO_MODELS,
    AGENTDOJO_SUBJECTS,
    DATA_SUBFOLDER_NAME,
)
from python_scripts.prompts import ALIGNMENTCHECK_USER_PROMPT_FORMAT
from llamafirewall import (
    AssistantMessage,
    Role,
    ScannerType,
    ScanResult,
    SystemMessage,
    ToolMessage,
    Trace,
    UserMessage,
    Message,
)

random.seed(42)


def get_normalized_key(sample_id: str):
    """Normalizes ID to handle different split depths in file paths."""
    parts = sample_id.split("/")
    return tuple(parts[-5:])


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


scanner_dict = {
    "alignmentcheck": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
    },
    "alignmentcheck_toolcheck": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.AGENT_ALIGNMENT],
    },
    "promptguard_toolcheck_alignmentcheck_normal": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD],
    },
    "promptguard_toolcheck_only_alignmentcheck_normal": {
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD],
    },
    "promptguard_toolcheck": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
        Role.TOOL: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
    },
    "promptguard_alignmentcheck": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD, ScannerType.AGENT_ALIGNMENT],
    },
    "promptguard_only_assistant": {
        Role.ASSISTANT: [ScannerType.PROMPT_GUARD],
    },
    "promptguard_only": {
        Role.USER: [ScannerType.PROMPT_GUARD],
        Role.SYSTEM: [ScannerType.PROMPT_GUARD],
    },
}


def log_scan_result(
    result, trace_path, optim_str, num_training_samples, model_name, idx, log_path
) -> None:
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            pass

    with open(log_path, "a") as f:
        log = {}
        log["optim_str"] = optim_str
        log["num_training_samples"] = num_training_samples
        log["trace_index"] = idx
        log["model_name"] = model_name
        log["trace_path"] = trace_path
        log["score"] = (f"{result.score:.4f}",)
        log["decision"] = str(result.decision)
        log["reason"] = str(result.reason)
        log["status"] = str(result.status)

        f.write(json.dumps(log, ensure_ascii=False) + "\n")


def get_trace(run, include_optim_str, optim_str, first_only, drop_tool_trace):
    trace = []
    for i in run["messages"]:
        if drop_tool_trace:
            if i.get("tool_calls"):
                del i["tool_calls"]
            if i["role"] == "tool":
                continue
        if include_optim_str:
            if isinstance(i["content"], str):
                i["content"] = insert_before_match(i["content"], optim_str, first_only)
            elif (
                i["content"]
                and i["content"][0]
                and isinstance(i["content"][0]["content"], str)
            ):
                i["content"][0]["content"] = insert_before_match(
                    i["content"][0]["content"], optim_str, first_only
                )
            elif i["content"] is None:
                continue
            else:
                raise ValueError("`content` field not found in trace.")
        message_content = json.dumps({k: v for k, v in i.items() if k != "role"})

        if i["role"] == "user":
            trace.append(UserMessage(content=message_content))
        elif i["role"] == "system":
            trace.append(SystemMessage(content=message_content))
        elif i["role"] == "assistant":
            trace.append(AssistantMessage(content=message_content))
        elif i["role"] == "tool":
            trace.append(ToolMessage(content=message_content))
        else:
            raise ValueError(f'Invalid role: {i["role"]}')

    return trace


def _pre_process_trace(message: Message, trace: Trace) -> str:
    return "\n".join([str(message) for message in trace + [message]])


def _pick_user_input(trace: Trace) -> str | None:
    for message in trace:
        if message.role == Role.USER:
            return message.content
    return None


def insert_before_match(main_string, optim_str, first_only):
    target_tag = "<INFORMATION>"
    replacement = optim_str + target_tag

    if first_only:
        return main_string.replace(target_tag, replacement, 1)

    return main_string.replace(target_tag, replacement)


def _get_user_prompt(trace) -> ScanResult:
    for current_ix, message in enumerate(trace):
        # Create a new trace up to the current message (excluding it)
        past_trace = [msg for ix, msg in enumerate(trace) if ix < current_ix]

    if past_trace is None:
        return

    user_input = _pick_user_input(past_trace)
    if not user_input:
        return

    trace_str = _pre_process_trace(message, past_trace)

    initial_user_prompt = ALIGNMENTCHECK_USER_PROMPT_FORMAT.format(
        user_message=user_input,
        trace=trace_str,
    )

    return initial_user_prompt


def get_user_prompts(
    base_path="./externals/agentdojo/runs",
    include_optim_str=True,
    first_only=False,
    optim_str="{optim_str}",
    return_string=True,
    security_target=True,
    list_of_models=AGENTDOJO_MODELS,
    run_file_patterns=None,
    max_length=18000,
    drop_tool_trace=False,
):
    if run_file_patterns is None:
        run_file_patterns = [
            f"{base_path}/{model}/{subject}/user_task_*/{attack}/*.json"
            for model, subject, attack in itertools.product(
                list_of_models, AGENTDOJO_SUBJECTS, AGENTDOJO_ATTACKS
            )
        ]

    matching_files = [f for i in run_file_patterns for f in glob.glob(i)]
    runs = []
    for file_path in matching_files:
        with open(file_path, "r") as f:
            run = json.load(f)
        run["trace_path"] = str(file_path)
        if run.get("security", "empty") == "empty":
            continue
        if security_target is None or (
            run.get("security", "empty") != "empty"
            and run["security"] == security_target
        ):
            runs.append(run)

    if security_target is None:
        insecure_runs = [i for i in runs if i["security"]]
        secure_runs = [i for i in runs if not i["security"]]

        min_len = min(len(insecure_runs), len(secure_runs))
        insecure_runs = random.sample(insecure_runs, k=min_len)
        secure_runs = random.sample(secure_runs, k=min_len)

        runs = insecure_runs + secure_runs

    final_runs = []
    for run in runs:
        temp_run = run.copy()
        if return_string:
            temp_run["messages"] = _get_user_prompt(
                get_trace(
                    run, include_optim_str, optim_str, first_only, drop_tool_trace
                )
            )
            if temp_run["messages"] and len(temp_run["messages"]) > max_length:
                continue
        else:
            temp_run["messages"] = get_trace(
                run, include_optim_str, optim_str, first_only, drop_tool_trace
            )
            if (
                _get_user_prompt(temp_run["messages"])
                and len(_get_user_prompt(temp_run["messages"])) > max_length
            ):
                continue
        final_runs.append(temp_run)

    return final_runs


def get_targets(model_id: str = DATA_SUBFOLDER_NAME):
    target_filepath = (
        f"./results/alignmentcheck/{model_id.replace('/', '_')}/generated_targets.jsonl"
    )

    targets = []
    with open(target_filepath, "r") as f:
        for line in f.readlines():
            target = json.loads(line.strip())
            targets.append(target)

    return targets


def get_metadata(run):
    if run.get("trace_metadata", False):
        metadata = run.get("trace_metadata")
    else:
        path = run["trace_path"]
        if not os.path.exists(path):
            path = "./externals/" + run["trace_path"]

        with open(path, "r") as f:
            metadata = json.load(f)

    return metadata


def get_existing_results(
    results_paths=glob.glob(
        f"results/alignmentcheck/{DATA_SUBFOLDER_NAME}/attacks_*.json"
    ),
    key=None,
    filter_=lambda x: True,
):
    seen = set()
    existing_results = []

    for path in results_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.decoder.JSONDecodeError:
                    print(line.strip())
                    print("*******************************")
                    print(path)
                    print("*******************************")

                if data.get("alignmentcheck_error", False):
                    continue
                if not key:
                    existing_results.append(data)
                    continue

                key_value = tuple(data[field] for field in key)
                if key_value not in seen:
                    seen.add(key_value)
                    existing_results.append(data)

    existing_results = list(filter(filter_, existing_results))
    return existing_results

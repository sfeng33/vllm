#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Post-run analysis of BFCL benchmark failures.

Reads BFCL score files, result files, and (optionally) vLLM tool-call
diagnostic logs to produce a structured diagnostic_report.json that
categorizes every failure by root cause.

Usage:
    python analyze-bfcl-failures.py \
        --result-dir ./result \
        --score-dir ./score \
        --diagnostic-log ./vllm_tool_call_diagnostic.jsonl \
        --output ./diagnostic_report.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import regex as re


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_score_files(score_dir: Path) -> list[dict]:
    failures = []
    for score_file in score_dir.rglob("*_score.json"):
        entries = load_jsonl(score_file)
        for entry in entries:
            if isinstance(entry, dict) and entry.get("valid") is False:
                failures.append(entry)
    return failures


def load_result_entries(result_dir: Path) -> dict[str, dict]:
    entries_by_id = {}
    for result_file in result_dir.rglob("*_result.json"):
        for entry in load_jsonl(result_file):
            if "id" in entry:
                entries_by_id[entry["id"]] = entry
    return entries_by_id


def load_diagnostic_log(log_path: Path | None) -> list[dict]:
    if log_path is None or not log_path.exists():
        return []
    return load_jsonl(log_path)


def _has_tool_call_patterns(raw_text: str) -> bool:
    """Heuristic: does the raw model output contain patterns that look like
    tool calls across common formats?"""
    patterns = [
        r'"name"\s*:\s*"',
        r'"arguments"\s*:\s*["{]',
        r"<tool_call>",
        r"<function=",
        r"functions\.",
        r"\[TOOL_CALLS\]",
        r">>>",
    ]
    return any(re.search(p, raw_text) for p in patterns)


def classify_failure(
    score_entry: dict,
    result_entry: dict | None,
    diagnostic_records: list[dict],
) -> dict:
    """Classify a single failure into a root-cause category."""
    test_id = score_entry.get("id", "unknown")
    error_type = score_entry.get("error_type", "")
    error_message = score_entry.get("error_message", score_entry.get("error", ""))

    model_result_raw = score_entry.get(
        "model_result_raw", score_entry.get("model_result")
    )
    model_result_decoded = score_entry.get("model_result_decoded")
    ground_truth = score_entry.get("possible_answer")
    inference_log = score_entry.get("inference_log", "")

    finish_reason = None
    message_content = None
    raw_model_output = None

    if result_entry:
        finish_reason = result_entry.get("finish_reason")
        message_content = result_entry.get("message_content")
        if not inference_log:
            inference_log = result_entry.get("inference_log", "")

    if diagnostic_records:
        raw_model_output = diagnostic_records[0].get("raw_model_output")

    category = _determine_category(
        error_type=error_type,
        error_message=error_message,
        model_result_raw=model_result_raw,
        model_result_decoded=model_result_decoded,
        ground_truth=ground_truth,
        finish_reason=finish_reason,
        raw_model_output=raw_model_output,
        inference_log=inference_log,
    )

    diagnosis = _build_diagnosis(
        category=category,
        error_type=error_type,
        error_message=error_message,
        model_result_raw=model_result_raw,
        model_result_decoded=model_result_decoded,
        ground_truth=ground_truth,
    )

    failure = {
        "test_id": test_id,
        "category": category,
        "error_type": error_type,
        "error_message": (
            error_message if isinstance(error_message, str) else str(error_message)
        ),
        "ground_truth": ground_truth,
        "model_result_raw": model_result_raw,
        "model_result_decoded": model_result_decoded,
        "finish_reason": finish_reason,
        "diagnosis": diagnosis,
    }

    if raw_model_output is not None:
        failure["raw_model_output"] = raw_model_output
    if message_content is not None:
        failure["message_content"] = message_content

    details = score_entry.get("error", {})
    if isinstance(details, dict) and "details" in details:
        failure["error_details"] = details["details"]
    elif isinstance(details, dict):
        failure["error_details"] = details

    return failure


def _determine_category(
    error_type: str,
    error_message,
    model_result_raw,
    model_result_decoded,
    ground_truth,
    finish_reason: str | None,
    raw_model_output: str | None,
    inference_log,
) -> str:
    if "force_terminated" in error_type:
        return "force_terminated"

    if "empty_turn_model_response" in error_type:
        return "empty_model_response"

    if "inference_error" in error_type:
        return "inference_error"

    err_str = str(error_message) if error_message else ""
    if "Error decoding" in err_str:
        return "decode_error"

    log_str = json.dumps(inference_log) if inference_log else ""
    if "Error decoding the model response" in log_str:
        return "decode_error"

    if (
        raw_model_output is not None
        and finish_reason
        and finish_reason != "tool_calls"
        and _has_tool_call_patterns(raw_model_output)
    ):
        return "vllm_parser_mismatch"

    if finish_reason and finish_reason != "tool_calls":
        return "model_no_tool_call"

    if "instance_state_mismatch" in error_type:
        return "execution_state_mismatch"

    if "execution_response_mismatch" in error_type:
        return "execution_response_mismatch"

    if "irrelevance_error" in error_type:
        return "model_irrelevance_error"

    if model_result_decoded is not None and ground_truth is not None:
        return _compare_tool_calls(model_result_decoded, ground_truth)

    if "decoder_failed" in error_type:
        return "decode_error"

    if "decoder_wrong_output_format" in error_type:
        return "model_wrong_format"

    return "unknown"


def _compare_tool_calls(model_decoded, ground_truth) -> str:
    """Compare decoded model output with ground truth to classify mismatch."""
    if not model_decoded and ground_truth:
        return "model_missing_calls"

    model_flat = _flatten_calls(model_decoded)
    gt_flat = _flatten_calls(ground_truth)

    if not model_flat and gt_flat:
        return "model_missing_calls"
    if model_flat and not gt_flat:
        return "model_extra_calls"

    model_names = {_extract_func_name(c) for c in model_flat}
    gt_names = {_extract_func_name(c) for c in gt_flat}

    if model_names != gt_names:
        if not model_names & gt_names:
            return "model_wrong_function"
        if len(model_flat) > len(gt_flat):
            return "model_extra_calls"
        if len(model_flat) < len(gt_flat):
            return "model_missing_calls"
        return "model_wrong_function"

    return "model_wrong_arguments"


def _flatten_calls(calls) -> list:
    if isinstance(calls, str):
        return [calls]
    if isinstance(calls, list):
        flat = []
        for item in calls:
            if isinstance(item, list):
                flat.extend(_flatten_calls(item))
            else:
                flat.append(item)
        return flat
    return [calls] if calls else []


def _extract_func_name(call) -> str:
    if isinstance(call, str):
        match = re.match(r"(\w+(?:\.\w+)*)\s*\(", call)
        if match:
            return match.group(1)
        return call
    if isinstance(call, dict):
        return next(iter(call.keys()), "")
    return str(call)


def _build_diagnosis(
    category: str,
    error_type: str,
    error_message,
    model_result_raw,
    model_result_decoded,
    ground_truth,
) -> str:
    msgs = {
        "model_no_tool_call": ("Model did not produce any tool call."),
        "model_wrong_function": ("Model called a different function."),
        "model_wrong_arguments": ("Model called correct function, wrong args."),
        "model_extra_calls": ("Model produced more tool calls than expected."),
        "model_missing_calls": ("Model produced fewer tool calls than expected."),
        "vllm_parser_mismatch": (
            "Raw output has tool call patterns but vLLM parser did not extract them."
        ),
        "decode_error": ("BFCL failed to decode model response."),
        "execution_state_mismatch": ("Calls executed but state differs from GT."),
        "execution_response_mismatch": ("Execution results don't match GT responses."),
        "force_terminated": ("Model exceeded the maximum step limit."),
        "empty_model_response": (
            "Model returned empty response for a turn requiring tool calls."
        ),
        "model_irrelevance_error": ("Model produced a call when it should not."),
        "inference_error": "Error during inference phase.",
        "model_wrong_format": ("Model output format doesn't match expected."),
        "unknown": "Unable to classify this failure.",
    }
    return msgs.get(category, f"Unclassified failure: {error_type}")


def _count_tests(result_dir: Path) -> int:
    total = 0
    for result_file in result_dir.rglob("*_result.json"):
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if "id" in entry:
                            total += 1
                    except json.JSONDecodeError:
                        pass
    return total


def analyze(
    result_dir: Path,
    score_dir: Path,
    diagnostic_log_path: Path | None,
    output_path: Path,
) -> dict:
    failures = load_score_files(score_dir)
    result_entries = load_result_entries(result_dir)
    diagnostic_records = load_diagnostic_log(diagnostic_log_path)

    diag_by_request_id: dict[str, list[dict]] = {}
    for rec in diagnostic_records:
        rid = rec.get("request_id", "")
        diag_by_request_id.setdefault(rid, []).append(rec)

    total_tests = _count_tests(result_dir)
    total_failed = len(failures)
    total_passed = total_tests - total_failed

    classified_failures = []
    category_counter: Counter = Counter()

    for score_entry in failures:
        test_id = score_entry.get("id", "unknown")
        result_entry = result_entries.get(test_id)

        matching_diag = []
        if diagnostic_records:
            matching_diag = [
                r
                for r in diagnostic_records
                if r.get("request_id", "").startswith(test_id[:20])
            ]

        failure = classify_failure(score_entry, result_entry, matching_diag)
        classified_failures.append(failure)
        category_counter[failure["category"]] += 1

    report = {
        "summary": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": (
                f"{total_passed / total_tests * 100:.2f}%" if total_tests > 0 else "N/A"
            ),
            "failure_breakdown": dict(category_counter.most_common()),
            "has_vllm_diagnostic_log": (len(diagnostic_records) > 0),
        },
        "failures": classified_failures,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Diagnostic report written to {output_path}")
    print(f"  Total: {total_tests}, Passed: {total_passed}, Failed: {total_failed}")
    if total_tests > 0:
        print(f"  Pass rate: {total_passed / total_tests * 100:.2f}%")
    print("  Failure breakdown:")
    for cat, count in category_counter.most_common():
        print(f"    {cat}: {count}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze BFCL benchmark failures")
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to BFCL result directory",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        required=True,
        help="Path to BFCL score directory",
    )
    parser.add_argument(
        "--diagnostic-log",
        type=Path,
        default=None,
        help="Path to vLLM tool call diagnostic JSONL log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_report.json"),
        help="Output path for the diagnostic report",
    )
    args = parser.parse_args()

    if not args.result_dir.exists():
        print(f"Error: result directory not found: {args.result_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.score_dir.exists():
        print(f"Error: score directory not found: {args.score_dir}", file=sys.stderr)
        sys.exit(1)

    analyze(args.result_dir, args.score_dir, args.diagnostic_log, args.output)


if __name__ == "__main__":
    main()

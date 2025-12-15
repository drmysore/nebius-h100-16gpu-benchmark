#!/usr/bin/env python3
"""
evaluate.py - Evaluate fine-tuned function calling model

Evaluates the model on:
1. Function call accuracy (correct function name)
2. Argument accuracy (correct argument values)
3. Response quality (when not calling functions)
4. Intent detection (call vs no-call decisions)

Usage:
    python evaluate.py \
        --model_path /mnt/shared/checkpoints/final \
        --test_data /mnt/shared/data/test.jsonl \
        --output_dir /mnt/shared/results
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    sample_id: int
    expected_function: Optional[str]
    predicted_function: Optional[str]
    function_match: bool
    expected_args: Optional[Dict]
    predicted_args: Optional[Dict]
    args_match: bool
    has_function_call: bool
    predicted_has_function_call: bool
    intent_match: bool
    raw_output: str


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_samples: int
    samples_with_fc: int
    samples_without_fc: int
    
    # Function calling metrics
    function_accuracy: float
    argument_accuracy: float
    intent_accuracy: float
    
    # Precision/Recall for function calls
    fc_precision: float
    fc_recall: float
    fc_f1: float
    
    # Per-function metrics
    per_function_accuracy: Dict[str, float]


def parse_function_call_from_output(output: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract function name and arguments from model output."""
    patterns = [
        # Standard format: <functioncall> {"name": ..., "arguments": ...}
        r'<functioncall>\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})',
        # Alternative: {"name": ..., "arguments": ...}
        r'\{"name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*(\{[^}]*\})\}',
        # Tool calls format
        r'"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*"([^"]+)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            func_name = match.group(1)
            try:
                args_str = match.group(2)
                # Handle escaped JSON
                if args_str.startswith('"') or '\\' in args_str:
                    args_str = args_str.replace('\\"', '"').replace('\\n', '\n')
                    if args_str.startswith('"'):
                        args_str = json.loads(args_str)
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                return func_name, args
            except json.JSONDecodeError:
                return func_name, None
    
    return None, None


def extract_expected_function_call(messages: List[Dict]) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract expected function call from reference messages."""
    for msg in messages:
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            func_info = tc.get("function", {})
            func_name = func_info.get("name")
            args_str = func_info.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = None
            return func_name, args
    return None, None


def compare_arguments(expected: Dict, predicted: Dict) -> bool:
    """Compare two argument dictionaries."""
    if expected is None and predicted is None:
        return True
    if expected is None or predicted is None:
        return False
    
    # Normalize and compare
    try:
        expected_str = json.dumps(expected, sort_keys=True).lower()
        predicted_str = json.dumps(predicted, sort_keys=True).lower()
        return expected_str == predicted_str
    except:
        return False


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    max_new_tokens: int = 512,
) -> str:
    """Generate model response for given messages."""
    # Build input without the last assistant message
    input_messages = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            break
        if msg["role"] == "assistant" and msg.get("content"):
            break
        if msg["role"] != "tool":  # Skip tool responses for input
            input_messages.append(msg)
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    sample_id: int,
) -> EvaluationResult:
    """Evaluate a single sample."""
    messages = sample.get("messages", [])
    has_function_call = sample.get("has_function_call", False)
    
    # Get expected function call
    expected_func, expected_args = extract_expected_function_call(messages)
    
    # Generate model response
    output = generate_response(model, tokenizer, messages)
    
    # Parse predicted function call
    predicted_func, predicted_args = parse_function_call_from_output(output)
    predicted_has_fc = predicted_func is not None
    
    # Calculate matches
    function_match = (expected_func == predicted_func)
    args_match = compare_arguments(expected_args, predicted_args) if function_match else False
    intent_match = (has_function_call == predicted_has_fc)
    
    return EvaluationResult(
        sample_id=sample_id,
        expected_function=expected_func,
        predicted_function=predicted_func,
        function_match=function_match,
        expected_args=expected_args,
        predicted_args=predicted_args,
        args_match=args_match,
        has_function_call=has_function_call,
        predicted_has_function_call=predicted_has_fc,
        intent_match=intent_match,
        raw_output=output,
    )


def compute_metrics(results: List[EvaluationResult]) -> EvaluationMetrics:
    """Compute aggregated metrics from results."""
    total = len(results)
    
    # Separate by function call presence
    fc_results = [r for r in results if r.has_function_call]
    non_fc_results = [r for r in results if not r.has_function_call]
    
    # Basic accuracy
    function_correct = sum(1 for r in fc_results if r.function_match)
    args_correct = sum(1 for r in fc_results if r.args_match)
    intent_correct = sum(1 for r in results if r.intent_match)
    
    function_accuracy = function_correct / len(fc_results) if fc_results else 0
    argument_accuracy = args_correct / len(fc_results) if fc_results else 0
    intent_accuracy = intent_correct / total if total else 0
    
    # Precision/Recall/F1 for function calls
    true_positives = sum(1 for r in results if r.has_function_call and r.predicted_has_function_call)
    false_positives = sum(1 for r in results if not r.has_function_call and r.predicted_has_function_call)
    false_negatives = sum(1 for r in results if r.has_function_call and not r.predicted_has_function_call)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per-function accuracy
    per_function = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in fc_results:
        if r.expected_function:
            per_function[r.expected_function]["total"] += 1
            if r.function_match:
                per_function[r.expected_function]["correct"] += 1
    
    per_function_accuracy = {
        func: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for func, stats in per_function.items()
    }
    
    return EvaluationMetrics(
        total_samples=total,
        samples_with_fc=len(fc_results),
        samples_without_fc=len(non_fc_results),
        function_accuracy=function_accuracy,
        argument_accuracy=argument_accuracy,
        intent_accuracy=intent_accuracy,
        fc_precision=precision,
        fc_recall=recall,
        fc_f1=f1,
        per_function_accuracy=per_function_accuracy,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate function calling model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = load_dataset("json", data_files=args.test_data, split="train")
    
    if args.max_samples:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
    
    logger.info(f"Evaluating {len(test_dataset)} samples")
    
    # Evaluate
    results = []
    for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        result = evaluate_sample(model, tokenizer, sample, i)
        results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Total samples: {metrics.total_samples}")
    logger.info(f"  - With function calls: {metrics.samples_with_fc}")
    logger.info(f"  - Without function calls: {metrics.samples_without_fc}")
    logger.info("-" * 60)
    logger.info(f"Function Accuracy: {metrics.function_accuracy:.2%}")
    logger.info(f"Argument Accuracy: {metrics.argument_accuracy:.2%}")
    logger.info(f"Intent Accuracy: {metrics.intent_accuracy:.2%}")
    logger.info("-" * 60)
    logger.info(f"FC Precision: {metrics.fc_precision:.2%}")
    logger.info(f"FC Recall: {metrics.fc_recall:.2%}")
    logger.info(f"FC F1: {metrics.fc_f1:.2%}")
    logger.info("=" * 60)
    
    # Save results
    results_data = [asdict(r) for r in results]
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    metrics_dict = {
        "total_samples": metrics.total_samples,
        "samples_with_fc": metrics.samples_with_fc,
        "samples_without_fc": metrics.samples_without_fc,
        "function_accuracy": metrics.function_accuracy,
        "argument_accuracy": metrics.argument_accuracy,
        "intent_accuracy": metrics.intent_accuracy,
        "fc_precision": metrics.fc_precision,
        "fc_recall": metrics.fc_recall,
        "fc_f1": metrics.fc_f1,
        "per_function_accuracy": metrics.per_function_accuracy,
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

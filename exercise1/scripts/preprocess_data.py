#!/usr/bin/env python3
"""
preprocess_data.py - Preprocess Glaive Function Calling dataset

This script:
1. Downloads the Glaive Function Calling v2 dataset
2. Cleans and validates the data
3. Converts to OpenAI-compatible chat format
4. Splits into train/val/test sets
5. Saves as JSONL files

Usage:
    python preprocess_data.py --output_dir /mnt/shared/data
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a function call."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """Represents a chat message."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


def extract_functions_from_system(system_content: str) -> List[Dict]:
    """Extract function definitions from system message."""
    functions = []
    
    # Try to find JSON function definitions
    json_pattern = r'\{[^{}]*"name"[^{}]*"parameters"[^{}]*\}'
    matches = re.findall(json_pattern, system_content, re.DOTALL)
    
    for match in matches:
        try:
            func_def = json.loads(match)
            if "name" in func_def:
                functions.append(func_def)
        except json.JSONDecodeError:
            continue
    
    return functions


def parse_function_call(content: str) -> Optional[FunctionCall]:
    """Parse a function call from assistant content."""
    # Pattern: <functioncall> {"name": "...", "arguments": {...}}
    patterns = [
        r'<functioncall>\s*(\{.*?\})\s*(?:</functioncall>)?',
        r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{.*?\})\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                if match.lastindex == 1:
                    # Full JSON match
                    data = json.loads(match.group(1))
                    return FunctionCall(
                        name=data.get("name", "unknown"),
                        arguments=data.get("arguments", {})
                    )
                elif match.lastindex == 2:
                    # Name and arguments separate
                    return FunctionCall(
                        name=match.group(1),
                        arguments=json.loads(match.group(2))
                    )
            except json.JSONDecodeError:
                continue
    
    return None


def parse_glaive_format(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse Glaive format into OpenAI-compatible format.
    
    Glaive format:
        SYSTEM: ...
        USER: ...
        ASSISTANT: <functioncall> {...}
        FUNCTION RESPONSE: {...}
        ASSISTANT: ...
    
    Output format:
        {
            "messages": [...],
            "tools": [...],
            "has_function_call": bool
        }
    """
    text = sample.get("chat", sample.get("conversations", ""))
    if not text or not isinstance(text, str):
        return None
    
    messages = []
    tools = []
    has_function_call = False
    tool_call_counter = 0
    
    # Split into segments
    segments = []
    current_segment = {"role": None, "content": ""}
    
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Check for role markers
        role_match = None
        for role_marker, role_name in [
            ("SYSTEM:", "system"),
            ("USER:", "user"),
            ("ASSISTANT:", "assistant"),
            ("FUNCTION RESPONSE:", "tool_response"),
        ]:
            if line.startswith(role_marker):
                if current_segment["role"]:
                    segments.append(current_segment)
                current_segment = {
                    "role": role_name,
                    "content": line[len(role_marker):].strip()
                }
                role_match = True
                break
        
        if not role_match and current_segment["role"]:
            current_segment["content"] += "\n" + line
    
    # Don't forget the last segment
    if current_segment["role"]:
        segments.append(current_segment)
    
    # Process segments into messages
    for segment in segments:
        role = segment["role"]
        content = segment["content"].strip()
        
        if role == "system":
            # Extract function definitions
            tools = extract_functions_from_system(content)
            messages.append({"role": "system", "content": content})
            
        elif role == "user":
            messages.append({"role": "user", "content": content})
            
        elif role == "assistant":
            # Check for function call
            func_call = parse_function_call(content)
            
            if func_call:
                has_function_call = True
                tool_call_id = f"call_{tool_call_counter}"
                tool_call_counter += 1
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(func_call.arguments) 
                                if isinstance(func_call.arguments, dict) 
                                else func_call.arguments
                        }
                    }]
                })
            else:
                # Regular assistant message
                if content:
                    messages.append({"role": "assistant", "content": content})
                    
        elif role == "tool_response":
            # Find the last tool call ID
            tool_call_id = f"call_{max(0, tool_call_counter - 1)}"
            for msg in reversed(messages):
                if msg.get("tool_calls"):
                    tool_call_id = msg["tool_calls"][0]["id"]
                    break
            
            messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id
            })
    
    # Validate: must have at least system and user message
    if len(messages) < 2:
        return None
    
    # Validate: must end with assistant message
    if messages[-1]["role"] not in ["assistant"]:
        return None
    
    return {
        "messages": messages,
        "tools": tools,
        "has_function_call": has_function_call
    }


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate a processed sample."""
    messages = sample.get("messages", [])
    
    # Basic validation
    if len(messages) < 2:
        return False
    
    # Check for required roles
    roles = [m["role"] for m in messages]
    if "system" not in roles and "user" not in roles:
        return False
    
    # Validate tool calls have corresponding responses
    pending_tool_calls = set()
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                pending_tool_calls.add(tc["id"])
        if msg["role"] == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id in pending_tool_calls:
                pending_tool_calls.remove(tc_id)
    
    # All tool calls should have responses
    if pending_tool_calls:
        return False
    
    return True


def process_dataset(
    dataset: Dataset,
    desc: str = "Processing"
) -> Dataset:
    """Process a dataset split."""
    processed = []
    
    for sample in tqdm(dataset, desc=desc):
        result = parse_glaive_format(sample)
        if result and validate_sample(result):
            processed.append(result)
    
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Glaive Function Calling dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/shared/data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for debugging)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading Glaive Function Calling v2 dataset...")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Process dataset
    logger.info("Processing dataset...")
    processed = process_dataset(dataset, "Processing")
    logger.info(f"Processed {len(processed)} valid samples")
    
    # Calculate statistics
    fc_count = sum(1 for s in processed if s["has_function_call"])
    logger.info(f"Samples with function calls: {fc_count} ({fc_count/len(processed)*100:.1f}%)")
    
    # Split dataset
    logger.info("Splitting dataset...")
    test_size = int(len(processed) * args.test_ratio)
    val_size = int(len(processed) * args.val_ratio)
    
    splits = processed.train_test_split(
        test_size=test_size + val_size,
        seed=args.seed
    )
    
    test_val = splits["test"].train_test_split(
        test_size=test_size / (test_size + val_size),
        seed=args.seed
    )
    
    train_dataset = splits["train"]
    val_dataset = test_val["train"]
    test_dataset = test_val["test"]
    
    logger.info(f"Train: {len(train_dataset)}")
    logger.info(f"Val: {len(val_dataset)}")
    logger.info(f"Test: {len(test_dataset)}")
    
    # Save datasets
    logger.info("Saving datasets...")
    
    train_dataset.to_json(output_dir / "train.jsonl")
    val_dataset.to_json(output_dir / "val.jsonl")
    test_dataset.to_json(output_dir / "test.jsonl")
    
    # Save metadata
    metadata = {
        "source": "glaiveai/glaive-function-calling-v2",
        "total_samples": len(processed),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "function_call_ratio": fc_count / len(processed),
        "seed": args.seed
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Data saved to {output_dir}")
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()

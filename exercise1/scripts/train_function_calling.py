#!/usr/bin/env python3
"""
train_function_calling.py - Multi-node distributed fine-tuning for function calling

This script fine-tunes an LLM for function calling using:
- DeepSpeed ZeRO-3 for efficient distributed training
- LoRA for parameter-efficient fine-tuning
- Flash Attention 2 for memory-efficient attention
- Gradient checkpointing for memory optimization

Usage:
    # Single node (8 GPUs)
    deepspeed --num_gpus=8 train_function_calling.py --config configs/training_config.yaml

    # Multi-node (via Slurm - see train_job.sbatch)
    srun deepspeed train_function_calling.py --config configs/training_config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import yaml
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Processing Functions
# =============================================================================

def parse_glaive_conversation(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Glaive function calling format into chat messages.
    
    Glaive format:
        SYSTEM: You are an helpful assistant...
        USER: user message
        ASSISTANT: <functioncall> {"name": "func", "arguments": {...}}
        FUNCTION RESPONSE: {"result": ...}
        ASSISTANT: response
    
    Output format (OpenAI-style):
        [{"role": "system", "content": "..."},
         {"role": "user", "content": "..."},
         {"role": "assistant", "content": null, "tool_calls": [...]},
         {"role": "tool", "content": "...", "tool_call_id": "..."},
         {"role": "assistant", "content": "..."}]
    """
    text = sample.get("chat", sample.get("conversations", ""))
    if isinstance(text, list):
        # Already in conversation format
        return {"messages": text}
    
    messages = []
    lines = text.strip().split("\n")
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for role markers
        if line.startswith("SYSTEM:"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = "system"
            current_content = [line[7:].strip()]
            
        elif line.startswith("USER:"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = "user"
            current_content = [line[5:].strip()]
            
        elif line.startswith("ASSISTANT:"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = "assistant"
            content = line[10:].strip()
            
            # Check for function call
            if "<functioncall>" in content:
                try:
                    # Extract function call JSON
                    fc_start = content.index("<functioncall>") + len("<functioncall>")
                    fc_json = content[fc_start:].strip()
                    if fc_json.endswith("</functioncall>"):
                        fc_json = fc_json[:-len("</functioncall>")]
                    
                    func_call = json.loads(fc_json)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{len(messages)}",
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", "unknown"),
                                "arguments": json.dumps(func_call.get("arguments", {}))
                            }
                        }]
                    })
                    current_role = None
                    current_content = []
                except (json.JSONDecodeError, ValueError):
                    current_content = [content]
            else:
                current_content = [content]
                
        elif line.startswith("FUNCTION RESPONSE:"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            # Add tool response
            tool_response = line[18:].strip()
            # Find the last tool call to get its ID
            tool_call_id = "call_0"
            for msg in reversed(messages):
                if msg.get("tool_calls"):
                    tool_call_id = msg["tool_calls"][0]["id"]
                    break
            
            messages.append({
                "role": "tool",
                "content": tool_response,
                "tool_call_id": tool_call_id
            })
            current_role = None
            current_content = []
            
        else:
            # Continuation of current role
            if current_role:
                current_content.append(line)
    
    # Don't forget the last message
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })
    
    return {"messages": messages}


def format_for_training(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
) -> Dict[str, Any]:
    """Format messages into training format using chat template."""
    messages = sample.get("messages", [])
    
    # Skip empty or invalid samples
    if not messages or len(messages) < 2:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    try:
        # Use tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
        
    except Exception as e:
        logger.warning(f"Error formatting sample: {e}")
        return {"input_ids": [], "attention_mask": [], "labels": []}


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load model and tokenizer with appropriate configurations."""
    model_config = config["model"]
    lora_config = config.get("lora", {})
    
    model_name = model_config["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
        padding_side="right",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        "use_cache": False,  # Disable for gradient checkpointing
    }
    
    # Flash attention
    if model_config.get("attn_implementation") == "flash_attention_2":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    # Apply LoRA if enabled
    if lora_config.get("enabled", True):
        logger.info("Applying LoRA configuration...")
        
        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=config["training"].get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            )
        )
    
    return model, tokenizer


# =============================================================================
# Dataset Loading Functions
# =============================================================================

def load_and_prepare_dataset(
    config: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> tuple:
    """Load and prepare training and validation datasets."""
    data_config = config["data"]
    
    # Load datasets
    logger.info("Loading datasets...")
    
    if data_config.get("train_file"):
        train_dataset = load_dataset(
            "json",
            data_files=data_config["train_file"],
            split="train",
        )
    else:
        # Load from HuggingFace
        dataset = load_dataset("glaiveai/glaive-function-calling-v2")
        splits = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = splits["train"]
    
    if data_config.get("validation_file"):
        eval_dataset = load_dataset(
            "json",
            data_files=data_config["validation_file"],
            split="train",
        )
    else:
        eval_dataset = splits["test"] if "splits" in dir() else None
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Parse conversations
    logger.info("Parsing conversations...")
    train_dataset = train_dataset.map(
        parse_glaive_conversation,
        num_proc=data_config.get("preprocessing_num_workers", 4),
        desc="Parsing train data",
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            parse_glaive_conversation,
            num_proc=data_config.get("preprocessing_num_workers", 4),
            desc="Parsing eval data",
        )
    
    # Format for training
    logger.info("Formatting for training...")
    max_length = data_config.get("max_seq_length", 4096)
    
    train_dataset = train_dataset.map(
        lambda x: format_for_training(x, tokenizer, max_length),
        num_proc=data_config.get("preprocessing_num_workers", 4),
        desc="Formatting train data",
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: format_for_training(x, tokenizer, max_length),
            num_proc=data_config.get("preprocessing_num_workers", 4),
            desc="Formatting eval data",
            remove_columns=eval_dataset.column_names,
        )
    
    # Filter empty samples
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering train data",
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids"]) > 0,
            desc="Filtering eval data",
        )
    
    logger.info(f"Final train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Final eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


# =============================================================================
# Training Arguments
# =============================================================================

def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Create TrainingArguments from config."""
    training_config = config["training"]
    output_config = config["output"]
    distributed_config = config.get("distributed", {})
    
    args = TrainingArguments(
        # Output
        output_dir=output_config["output_dir"],
        logging_dir=output_config.get("logging_dir"),
        overwrite_output_dir=output_config.get("overwrite_output_dir", False),
        
        # Training
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        
        # Learning rate
        learning_rate=training_config.get("learning_rate", 2e-5),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        
        # Optimization
        optim=training_config.get("optim", "adamw_torch_fused"),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Precision
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        tf32=training_config.get("tf32", True),
        
        # Gradient checkpointing
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=training_config.get("logging_first_step", True),
        report_to=training_config.get("report_to", ["tensorboard"]),
        
        # Saving
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        save_safetensors=training_config.get("save_safetensors", True),
        
        # Evaluation
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        
        # Misc
        seed=training_config.get("seed", 42),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        remove_unused_columns=training_config.get("remove_unused_columns", True),
        ddp_find_unused_parameters=training_config.get("ddp_find_unused_parameters", False),
        
        # Distributed
        deepspeed=distributed_config.get("deepspeed"),
        fsdp=distributed_config.get("fsdp", ""),
        fsdp_config=distributed_config.get("fsdp_config"),
        ddp_backend=distributed_config.get("ddp_backend", "nccl"),
        local_rank=distributed_config.get("local_rank", -1),
    )
    
    return args


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for function calling")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Update local_rank from args
    if args.local_rank != -1:
        config.setdefault("distributed", {})["local_rank"] = args.local_rank
    
    # Set seed
    set_seed(config["training"].get("seed", 42))
    
    # Log configuration
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        logger.info("=" * 60)
        logger.info("LLM Function Calling Fine-tuning")
        logger.info("=" * 60)
        logger.info(f"Configuration:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load datasets
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if rank == 0:
        logger.info("Starting training...")
    
    # Resume from checkpoint if specified
    resume_from = config["output"].get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Save final model
    if rank == 0:
        logger.info("Saving final model...")
        final_path = Path(config["output"]["output_dir"]) / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        
        # Save training config
        with open(final_path / "training_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        logger.info(f"Model saved to {final_path}")
        logger.info("Training completed!")


if __name__ == "__main__":
    main()

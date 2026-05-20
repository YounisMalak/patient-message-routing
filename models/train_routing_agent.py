"""
Training Script for Routing Agent

Fine-tunes Llama-3.2-3B-Instruct using LoRA for message routing.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

from config import ROUTING_MODEL_CONFIG, LABEL_TO_TOKEN


def load_system_prompt(prompt_path: str) -> str:
    """Load system prompt from file."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_training_data(data_path: str) -> list:
    """Load training data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_example(example: dict, system_prompt: str, tokenizer) -> dict:
    """
    Format a single example for training.
    
    Creates chat-format input with system prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['text']},
        {"role": "assistant", "content": LABEL_TO_TOKEN[example['label']]}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def train_routing_agent(
    train_data_path: str,
    system_prompt_path: str,
    output_dir: str,
    base_model: str = None,
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    seed: int = None,
):
    """
    Train the routing agent using LoRA fine-tuning.
    
    Args:
        train_data_path: Path to training data JSON
        system_prompt_path: Path to system prompt file
        output_dir: Directory to save fine-tuned model
        base_model: Base model name (default from config)
        num_epochs: Number of training epochs (default from config)
        batch_size: Per-device batch size (default from config)
        learning_rate: Learning rate (default from config)
        seed: Random seed (default from config)
    """
    # Use config defaults if not specified
    config = ROUTING_MODEL_CONFIG
    base_model = base_model or config["base_model"]
    num_epochs = num_epochs or config["num_epochs"]
    batch_size = batch_size or config["per_device_batch_size"]
    learning_rate = learning_rate or config["learning_rate"]
    seed = seed or config["seed"]
    
    print("=" * 60)
    print("TRAINING ROUTING AGENT")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)
    print(f"Loaded system prompt: {len(system_prompt)} chars")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    print("Loading training data...")
    train_data = load_training_data(train_data_path)
    print(f"Loaded {len(train_data)} training examples")
    
    # Format examples
    print("Formatting examples...")
    formatted_data = [
        format_example(ex, system_prompt, tokenizer)
        for ex in train_data
    ]
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_seq_length"]),
        batched=True,
        remove_columns=["text"],
    )
    
    # Add labels (same as input_ids for causal LM)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
        device_map="auto",
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save system prompt with model
    with open(os.path.join(output_dir, "system_prompt.txt"), 'w', encoding='utf-8') as f:
        f.write(system_prompt)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train routing agent")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--system_prompt", type=str, 
                        default="../ontology/system_prompt.txt",
                        help="Path to system prompt file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model (default: Llama-3.2-3B-Instruct)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train_routing_agent(
        train_data_path=args.train_data,
        system_prompt_path=args.system_prompt,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

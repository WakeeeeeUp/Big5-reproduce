# scripts/train_trait_lora.py
# Transformers == 4.31.*, TRL == 0.4.7
# Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Scheme A: single LoRA + trait tokens injected into the tokenizer.

import os
import json
import argparse
import shutil
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATA_PATH = "data/prepared/trait_cond_sft.jsonl"
DEFAULT_TRAITS_JSON = "data/prepared/traits.json"
DEFAULT_OUT_DIR = "expert_ckpt/trait_cond_lora"


def _load_trait_tokens(traits_json_path):
    with open(traits_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    trait_tokens = d.get("trait_tokens", [])
    end_token = d.get("end_token")
    return trait_tokens, end_token


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--traits_json", default=DEFAULT_TRAITS_JSON)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=1024)

    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------
    # Tokenizer + trait tokens
    # --------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Ensure EOS/PAD exist for batching with Trainer on LLaMA-like tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trait_tokens, end_token = _load_trait_tokens(args.traits_json)
    to_add = list({*(trait_tokens or []), *( [end_token] if end_token else [] )} - set(tokenizer.get_vocab().keys()))
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    # --------------------
    # Model (optional 4-bit)
    # --------------------
    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        quantization_config=quant_cfg,
    )

    # Resize embeddings if we added special tokens
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    # --------------------
    # Dataset
    # Expect fields: {"text": <prompt_with_trait_tokens>, "labels": <assistant_reply>}
    # We'll stitch them together via formatting_func to make a single training string.
    # --------------------
    ds = load_dataset("json", data_files=args.data_path, split="train")

    def formatting_func(examples):
        # TRL 0.4.7 requires a list of strings returned.
        texts = []
        eos = tokenizer.eos_token or ""
        for t, y in zip(examples["text"], examples.get("labels", [""] * len(examples["text"]))):
            texts.append((t or "") + (y or "") + eos)
        return texts

    # --------------------
    # LoRA config (PEFT)
    # --------------------
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --------------------
    # TrainingArguments (Transformers 4.31)
    # Autoselect precision: bf16 on Ampere+, otherwise fp16 if supported.
    # --------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=2,
        optim="adamw_torch",          # conservative choice for 4.31 + TRL 0.4.7
        report_to=[],
        remove_unused_columns=False,   # important for TRL SFTTrainer
    )

    # --------------------
    # SFTTrainer (TRL 0.4.7)
    # --------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field=None,         # we use formatting_func instead
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,                   # safest for 0.4.7
        args=training_args,
        peft_config=lora_cfg,            # TRL will wrap the model with PEFT/LoRA
    )

    trainer.train()

    # --------------------
    # Save LoRA adapter + tokenizer + traits.json
    # --------------------
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    try:
        shutil.copyfile(args.traits_json, os.path.join(args.out_dir, "traits.json"))
    except Exception as e:
        print(f"Warning: failed to copy traits.json: {e}")

    print("Saved LoRA and tokenizer to:", args.out_dir)


if __name__ == "__main__":
    main()

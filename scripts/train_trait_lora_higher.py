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

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import hf_hub_download


DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
DEFAULT_OUT_DIR = "expert_ckpt_higher/trait_cond_lora"
REPO_ID = "Eden-D/big5-traits"

DEFAULT_DATA_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/prepared/trait_cond_sft.jsonl",
    repo_type="dataset",
)

DEFAULT_TRAITS_JSON = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/prepared/traits.json",
    repo_type="dataset",
)


def _load_trait_tokens(traits_json_path: str):
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

    # 1x16GB-friendly defaults (QLoRA)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=1024)

    # QLoRA switch
    ap.add_argument("--use_4bit", action="store_true", help="Enable 4-bit QLoRA (recommended for 1x16GB).")
    ap.add_argument("--trust_remote_code", action="store_true", help="Only if your model repo needs it.")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Optional: override target modules
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list for LoRA target modules.",
    )

    # Training misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_strategy", type=str, default="epoch")  # "epoch" or "steps"
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable grad checkpointing (recommended).")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # a bit faster on Ampere+ without affecting training quality much
        torch.backends.cuda.matmul.allow_tf32 = True

    # --------------------
    # Tokenizer + trait tokens
    # --------------------
    tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    use_fast=False,   # LLaMA-2 在旧 transformers 下更稳
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Ensure EOS/PAD exist for batching
    if tokenizer.eos_token is None:
        # Rare, but just in case
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trait_tokens, end_token = _load_trait_tokens(args.traits_json)

    vocab = tokenizer.get_vocab().keys()
    to_add_set = set(trait_tokens or [])
    if end_token:
        to_add_set.add(end_token)

    # only add tokens not in vocab
    to_add = [t for t in to_add_set if t not in vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    # --------------------
    # Model (optional 4-bit QLoRA)
    # --------------------
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype if torch.cuda.is_available() else "auto",
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )

    # Resize embeddings if we added special tokens
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    # Training-time memory / stability tweaks
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # QLoRA prep (important!)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # --------------------
    # Dataset
    # Expect fields: {"text": <prompt_with_trait_tokens>, "labels": <assistant_reply>}
    # We'll stitch them together via formatting_func to make a single training string.
    # --------------------
    ds = load_dataset("json", data_files=args.data_path, split="train")

    def formatting_func(examples):
        texts = []
        eos = tokenizer.eos_token or ""
        labels = examples.get("labels", [""] * len(examples["text"]))
        for t, y in zip(examples["text"], labels):
            texts.append((t or "") + (y or "") + eos)
        return texts

    # --------------------
    # LoRA config (PEFT)
    # --------------------
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # --------------------
    # TrainingArguments
    # --------------------
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=use_bf16,
        fp16=(not use_bf16) if torch.cuda.is_available() else False,
        dataloader_num_workers=2,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,  # important for TRL SFTTrainer
    )

    # --------------------
    # SFTTrainer (TRL)
    # --------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field=None,      # we use formatting_func instead
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
        peft_config=lora_cfg,
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

import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
import torch
from peft import LoraConfig
from trl import SFTTrainer

DATA = "./data/psych_alpaca.jsonl"
OUT_DIR = "./expert_ckpt"

MODEL_NAME = "tiiuae/falcon-7b"
USE_QLORA  = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb = None
load_dtype = "auto"
if USE_QLORA:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
else:
    load_dtype = "bfloat16"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16   # 或 bfloat16
)


# LoRA 配置
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# 载入 jsonl 数据为 HF 数据集
ds = load_dataset("json", data_files=DATA, split="train")

def format_example(ex):
    # Alpaca 格式拼接
    inst = ex["instruction"].strip()
    inp  = ex.get("input","").strip()
    out  = ex["output"].strip()
    if inp:
        prompt = f"Instruction: {inst}\nInput: {inp}\nAnswer:"
    else:
        prompt = f"Instruction: {inst}\nAnswer:"
    return {"text": prompt + " " + out}

ds = ds.map(format_example, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # 有效batch=32
    learning_rate=2e-4,              # LoRA/QLoRA较大学习率常见
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=ds,
    peft_config=peft_cfg,
    # max_seq_length=1024,
    # dataset_text_field="text",
    # packing=True,
    args=args,
)

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved:", OUT_DIR)

# -*- coding: utf-8 -*-
"""
Compatible SFT training script:
- 优先使用 TRL.SFTTrainer（若缺失则回退到 transformers.Trainer）
- 自动处理 tokenizer 参数是否可传
- 自动关闭 bf16/fp16（按设备支持）
- bitsandbytes(QLoRA) 可选：若不可用则改为常规加载
"""

import os, sys, json
import importlib
import torch

# ---------- 可配置 ----------
DATA = "./data/psych_alpaca.jsonl"     # 脚本1生成的 SFT 数据（Alpaca格式）
OUT_DIR = "./expert_ckpt"              # 输出目录
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_QLORA = True                       # 若无GPU或bitsandbytes不可用，会自动降级
MAX_SEQ_LEN = 1024
EPOCHS = 3
LR = 2e-4
BATCH = 2
GRAD_ACC = 16

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 依赖探测 ----------
def has_module(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

has_trl = has_module("trl")
has_peft = has_module("peft")
has_bnb = has_module("bitsandbytes")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 量化配置（可选）
quant_args = {}
if USE_QLORA and has_bnb and torch.cuda.is_available():
    from transformers import BitsAndBytesConfig
    quant_args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
else:
    # 不可量化就空着
    pass

# torch 精度：自动判断
precision_kwargs = {}
use_cuda = torch.cuda.is_available()
if use_cuda:
    # 优先 bf16，其次 fp16
    if torch.cuda.is_bf16_supported():
        precision_kwargs["bf16"] = True
    else:
        precision_kwargs["fp16"] = True
# CPU/MPS: 不设混合精度参数

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    **quant_args
)

# 准备数据（把 alpaca 字段拼成 text）
ds = load_dataset("json", data_files=DATA, split="train")
def format_example(ex):
    inst = ex["instruction"].strip()
    inp  = (ex.get("input") or "").strip()
    out  = ex["output"].strip()
    if inp:
        prompt = f"Instruction: {inst}\nInput: {inp}\nAnswer:"
    else:
        prompt = f"Instruction: {inst}\nAnswer:"
    return {"text": prompt + " " + out}
ds = ds.map(format_example, remove_columns=ds.column_names)

# 训练参数：兼容老/新版 transformers
def make_training_args():
    kwargs = dict(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",      # 旧版不报错；若极旧可删
        report_to="none",
        **precision_kwargs
    )
    # evaluation_strategy 一律不设（兼容老版）
    return TrainingArguments(**kwargs)

args = make_training_args()

# 优先走 TRL.SFTTrainer
if has_trl and has_peft:
    from peft import LoraConfig
    from trl import SFTTrainer

    # LoRA 配置
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    # 兼容：部分旧版 SFTTrainer 不接收 tokenizer 参数 -> 尝试后回退
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            peft_config=peft_cfg,
            # max_seq_length=MAX_SEQ_LEN,
            # dataset_text_field="text",
            # packing=True,
            args=args,
        )
    except TypeError:
        # 去掉 tokenizer 参数再试
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            peft_config=peft_cfg,
            # max_seq_length=MAX_SEQ_LEN,
            # dataset_text_field="text",
            # packing=True,
            args=args,
        )
else:
    # 回退到纯 transformers.Trainer（无LoRA）
    from transformers import Trainer, DataCollatorForLanguageModeling

    # tokenize
    def tok_map(ex):
        return tokenizer(
            ex["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN
        )
    ds_tok = ds.map(tok_map, batched=True, remove_columns=ds.column_names)
    ds_tok.set_format(type="torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 兼容：旧版 Trainer 可能不接受 tokenizer 参数
    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_tok,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_tok,
            data_collator=data_collator,
        )

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved:", OUT_DIR)
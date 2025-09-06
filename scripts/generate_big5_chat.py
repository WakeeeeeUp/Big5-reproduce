# scripts/generate_big5_chat_gen.py
import os, sys, json, torch
from pathlib import Path
from tqdm import tqdm

# ---- 1) 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 这些模块在 llm_personality 子目录下
from llm_personality.profile_creation.personality_prompts import personality_descriptions as TRAIT_TO_DESC
from llm_personality.profile_creation.dialogue_utils import build_turn0_from_soda

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# ---- 2) 配置：可用环境变量覆盖，否则用公开模型作为默认 ----
BASE_LM = os.environ.get(
    "BASE_LM",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 公开、可直接下
)

# 你的 LoRA/QLoRA 适配器（用脚本2训练后产生），可以是目录或每个 trait 一个子目录
EXPERT = os.environ.get("EXPERT", "./expert_ckpt")

# 场景+Turn0+trait 的输入（每行一个 json：{"scenario":..., "participants":..., "trait":"agreeableness-high", ...}）
CONTEXTS = os.environ.get("CONTEXTS", "./data/soda_contexts.jsonl")

# 生成输出路径
OUT_PATH = os.environ.get("OUT_PATH", "./generations/big5_chat_gen.jsonl")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ---- 3) 设备选择：CUDA> MPS > CPU，并按设备选择量化/精度 ----
if torch.cuda.is_available():
    device = "cuda"
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_LM, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_LM,
        device_map="auto",
        quantization_config=bnb_cfg
    )
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    tokenizer = AutoTokenizer.from_pretrained(BASE_LM, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_LM,
        torch_dtype=torch.float16
    ).to("mps")
else:
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_LM, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_LM)

print(f"[Info] device={device}, basemodel={BASE_LM}")

# ---- 4) 载入 LoRA/QLoRA 适配器（expert） ----
# 如果你把所有 trait 的 adapter 都放在 expert_ckpt/ 里，按需切换目录即可。
# 也支持把单一 adapter 放在 EXPERT 指向的目录。
def load_expert_adapter(base_model, expert_dir: str):
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"Expert adapter not found: {expert_dir}")
    model = PeftModel.from_pretrained(base_model, expert_dir)
    model.eval()
    return model

# 下面是一个最小生成流程示例，确保 import/设备/加载都没问题后再接你的 dexperts_blend_logits 逻辑：
def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    # 示例：如果你每个 trait 有一个子目录：expert_ckpt/agreeableness-high 等
    # 这里按行读取 CONTEXTS 的 trait 字段，动态切换 LoRA
    current_trait = None
    lora_model = None

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for ex in tqdm(iter_jsonl(CONTEXTS)):
            trait = ex.get("trait")  # e.g. "agreeableness-high"
            if trait != current_trait:
                # 切换/加载对应的 LoRA 目录
                expert_dir = os.path.join(EXPERT, trait)
                lora_model = load_expert_adapter(base_model, expert_dir)
                current_trait = trait

            # 构建 Turn#0（如果你已有 turn0 文本就跳过这步）
            turn0 = build_turn0_from_soda(
                scenario=ex["scenario"],
                participants=ex["participants"],
                persona_desc=TRAIT_TO_DESC[trait]
            )

            # 你的 dexpert 混合策略（若不用可直接 lora_model.generate）
            # logits = dexperts_blend_logits(...)
            # 这里给个直出示例：
            inputs = tokenizer(turn0, return_tensors="pt").to(lora_model.device)
            out_ids = lora_model.generate(**inputs, max_new_tokens=80, do_sample=True, top_p=0.9)
            text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            ex_out = {**ex, "turn0": turn0, "gen": text}
            fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

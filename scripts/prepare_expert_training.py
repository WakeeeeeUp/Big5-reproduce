import json, os, random
from tqdm import tqdm

random.seed(42)

SRC_HF_DIR = "./hf_big5_dataset"  # 也可以换成你自建的 PsychGenerator JSONL
OUT_JSONL  = "./data/psych_alpaca.jsonl"

# 你如果没有 PsychGenerator，就用你现有的 big5_chat 数据构造 SFT对
# 规则：instruction 带 trait，input=前5词，output=剩余正文
# 注意：这是一种“代理方案”，与论文的 PsychGenerator 完全一致度有限，但流程吻合。

# 从HF数据集读取：每条包含 ['instruction','input','output','trait', ...]
from datasets import load_from_disk
ds = load_from_disk(SRC_HF_DIR)

TRAITS = ["agreeableness","openness","conscientiousness","extraversion","neuroticism"]
KEEP = []
for t in TRAITS:
    sub = ds.filter(lambda x: x["trait"].startswith(f"{t}-"))
    # 均衡采样 high/low
    highs = [ex for ex in sub if ex["trait"].endswith("high")]
    lows  = [ex for ex in sub if ex["trait"].endswith("low")]
    n = min(len(highs), len(lows))
    highs = highs[:n]
    lows  = lows[:n]
    KEEP.extend(highs + lows)

def first5(s):
    toks = s.strip().split()
    return " ".join(toks[:5])

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for ex in tqdm(KEEP, desc="build alpaca"):
        trait = ex["trait"].split("-")[0].capitalize()
        level = ex["trait"].split("-")[1]
        # 构造 instruction
        inst = f"Complete this post with Big Five Personality: {trait} - {level}."
        # 取 input 的前5词作为“提示”（若不存在input就用prompt里Turn #0的内容截取）
        inp = ex.get("input","") or ex.get("prompt","")
        inp = first5(inp)

        out = ex["output"].strip()
        if not out: 
            continue

        f.write(json.dumps({
            "instruction": inst,
            "input": inp,
            "output": out,
            "trait": ex["trait"]
        }, ensure_ascii=False) + "\n")

print("Saved:", OUT_JSONL)

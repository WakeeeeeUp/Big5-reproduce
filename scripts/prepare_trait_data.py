# scripts/prepare_trait_data.py
import json, os, argparse
from collections import OrderedDict

SPECIAL_END = "<|traits_end|>"


TRAITS = [
"agreeableness-high", "agreeableness-low",
"conscientiousness-high", "conscientiousness-low",
"extraversion-high", "extraversion-low",
"neuroticism-high", "neuroticism-low",
"openness-high", "openness-low",
]

def make_trait_token(trait: str) -> str:
    return f"<|trait:{trait}|>"

def build_prompt(instruction: str, user_input: str, trait: str) -> str:
    trait_tok = make_trait_token(trait)
    system = (
        "You are a helpful assistant. Adopt the following Big Five traits: "
        f"{trait_tok} {SPECIAL_END}"
    )
    user = instruction if instruction else ""
    if user_input: 
        user = (user + "\n" + user_input).strip()
    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system +
        "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + user +
        "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/psych_alpaca.jsonl")
    ap.add_argument("--out_path", default="data/prepared/trait_cond_sft.jsonl")
    args = ap.parse_args()
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    seen_traits = OrderedDict()
    cnt = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:
             for line in fin:
                ex = json.loads(line)
                trait = ex.get("trait")
                if not trait:
                    continue
                seen_traits[trait] = 1
                text = build_prompt(ex.get("instruction", ""), ex.get("input", ""), trait)
                out = {
                    "text": text,
                    "labels": ex.get("output", ""),
                    "trait": trait,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                cnt += 1
                
    traits_json_path = os.path.join(os.path.dirname(args.out_path), "traits.json")
    trait_tokens = [make_trait_token(t) for t in seen_traits.keys()]
    with open(traits_json_path, "w", encoding="utf-8") as f:
        json.dump({"trait_tokens": trait_tokens, "end_token": SPECIAL_END}, f, ensure_ascii=False, indent=2)
        
    print(f"Wrote {cnt} examples → {args.out_path}")
    print(f"Trait tokens → {traits_json_path}")
    
if __name__ == "__main__":
    main()
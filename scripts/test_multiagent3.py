from __future__ import annotations
from typing import TypedDict, Literal, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

"""
Affective & Pedagogical Multi‑Agent Peer System (Single‑Trait Agent)

This script implements the three agents described in the RTF spec:
1) PersonalityAlignmentAgent – the experimental modulator that ONLY applies linguistic style
   (baseline | alignment | contradictory) using the user's Big Five profile.
2) MotivationalPeerAgent – generates ARCS‑grounded motivational CONTENT in neutral style.
3) PeerGuidelineAgent – formats the response for clarity, goal alignment, and pacing.

Design principles from the spec:
- PersonalityAlignmentAgent must NOT change motivational/pedagogical intent; it changes delivery style only.
- MotivationalPeerAgent applies ARCS (chooses the most suitable focus per turn) and stays style‑neutral.
- PeerGuidelineAgent ensures structure/pacing stays aligned with the immediate task/goal.
- The LoRA adapter (trait_cond_lora) is loaded and used for all generations.
- Avoid hard‑coded heuristics and canned phrases; rely on the model + prompts.
"""

# ============================
# Model loading (adapter)
# ============================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "Eden-D/trait_cond_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# Simple text generation helper (single‑turn)
def llm_generate(prompt: str, max_new_tokens: int = 384) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================
# Types
# ============================
Condition = Literal["baseline", "alignment", "contradictory"]

class ConversationState(TypedDict, total=False):
    user_input: str
    condition: Condition
    user_traits: Dict[str, int]  # Big Five: 1–100

    # Produced artifacts
    style_sheet: str             # personality style guidance (markdown bullets)
    motivational_content: str    # ARCS content (neutral style)
    structured_response: str     # pedagogically formatted (neutral style)
    final_response: str          # styled + structured

# ============================
# Agents
# ============================
@dataclass
class PersonalityAlignmentAgent:
    """Linguistic/affective style modulator.
    IMPORTANT: Does not change ARCS intent; only delivery style.
    """

    @staticmethod
    def _bucket(v: int) -> str:
        return "low" if v <= 40 else ("moderate" if v <= 60 else "high")

    @staticmethod
    def _invert_bucket(b: str) -> str:
        return {"low": "high", "high": "low"}.get(b, "moderate")

    def style_sheet(self, traits: Dict[str, int], condition: Condition) -> str:
        buckets = {k: self._bucket(int(v)) for k, v in (traits or {}).items()}
        if condition == "contradictory":
            buckets = {k: self._invert_bucket(b) for k, b in buckets.items()}

        rtf_role = """### SYSTEM ROLE
You are the linguistic and affective modulator. Apply the designated personality style (baseline | aligned | contradictory) to the motivational output. Do NOT alter ARCS intent.
"""
        prompt = f"""
{rtf_role}
## Condition
{condition}

## Trait buckets (low/moderate/high)
{buckets}

## Task
Generate a concise style sheet (markdown bullets) specifying tone, word choice, sentence length, assertiveness, and pacing appropriate to the condition and buckets. Keep it compact and actionable; avoid repeating user content.
"""
        return llm_generate(prompt)

    def stylize(self, base_text: str, style_sheet: str) -> str:
        rtf_constraint = (
            "### CONSTRAINT\n"
            "You must preserve the motivational and pedagogical meaning; only adjust delivery style.\n"
        )
        prompt = f"""{rtf_constraint}
## Style Sheet
{style_sheet}

## Text to Style (do not change intent)
{base_text}

## Output
Produce the same content, rewritten to follow the style sheet.
"""
        return llm_generate(prompt)


@dataclass
class MotivationalPeerAgent:
    """Generates ARCS‑based motivational content in a neutral style.
    The PersonalityAlignmentAgent will handle stylistic delivery after this.
    """

    def run(self, user_input: str) -> str:
        role = """### SYSTEM ROLE
You are a motivational AI peer and classmate. You and the user are equals and collaborators. Apply the ARCS model to maintain and enhance motivation and confidence. Respond in a neutral, unstyled voice; style is applied later.
"""
        prompt = f"""{role}
## User Message
{user_input}

## Task
Select the most helpful single ARCS focus for this turn (A/R/C/S) and compose motivational content that supports the user's next step. Keep it concise and actionable.

## Output Format
Write plain text (no JSON) suitable for a peer chat. Avoid fixed templates and clichés.
"""
        return llm_generate(prompt)


@dataclass
class PeerGuidelineAgent:
    """Pedagogical backbone: structure, goal alignment, pacing.
    Works in neutral style; final styling applied by PersonalityAlignmentAgent.
    """

    def run(self, user_input: str, motivational_text: str) -> str:
        role = """### SYSTEM ROLE
You are the structural and goal‑setting component of the peer team. Keep collaboration efficient, focused, and goal‑directed. Ensure motivational content links back to specific learning tasks and objectives.
"""
        prompt = f"""{role}
## Inputs
User Message:
{user_input}

Motivational Content:
{motivational_text}

## Task
1) Link motivational content back to the user's immediate goal,
2) Propose a small next step or two,
3) Include a quick check‑for‑understanding question.
Write in neutral style; avoid rigid templates.
"""
        return llm_generate(prompt)


class Orchestrator:
    """Coordinates the three agents: alignment (style), motive (ARCS content), and guide (structure)."""

    def __init__(self) -> None:
        self.align = PersonalityAlignmentAgent()
        self.motive = MotivationalPeerAgent()
        self.guide = PeerGuidelineAgent()

    def step(self, state: ConversationState) -> ConversationState:
        # 1) Produce a style sheet for this user + condition
        style = self.align.style_sheet(state.get("user_traits", {}), state.get("condition", "baseline"))
        state["style_sheet"] = style

        # 2) Generate ARCS motivational content (neutral style)
        mot = self.motive.run(state.get("user_input", ""))
        state["motivational_content"] = mot

        # 3) Structure and goal‑align (neutral style)
        structured = self.guide.run(state.get("user_input", ""), mot)
        state["structured_response"] = structured

        # 4) Apply final personality styling (without changing intent)
        styled = self.align.stylize(structured, style)
        state["final_response"] = styled
        return state

# ============================
# Example usage
# ============================
if __name__ == "__main__":
    orchestrator = Orchestrator()

    demo_state: ConversationState = {
        "user_input": "I'm stuck organizing a literature review and not sure what a solid next step looks like.",
        "condition": "alignment",   # baseline | alignment | contradictory
        "user_traits": {
            "Extraversion": 55,
            "Agreeableness": 40,
            "Conscientiousness": 80,
            "Openness": 35,
            "Neuroticism": 45,
        },
    }

    out = orchestrator.step(demo_state)
    print(out["final_response"])
    
    print("----- FINAL RESPONSE -----")
    print(out["final_response"])
    print("----- END -----")

"""Builds the four prompts for one task."""
from modules.prompt_templates import TEMPLATES, FEW_SHOT_EXAMPLES, DEFAULT_ROLES
from config import TECHNIQUES


def format_examples(category: str, n: int) -> str:
    pool = FEW_SHOT_EXAMPLES.get(category, [])[:max(0, n)]
    return "\n\n".join(f"Task: {q}\nAnswer: {a}" for q, a in pool) or "(no examples available)"


def build_prompts(task: str, category: str, n_examples: int = 2, role: str = None,
                  templates: dict = None) -> dict:
    templates = templates or TEMPLATES
    role = role or DEFAULT_ROLES.get(category, "an expert")
    out = {}
    for t in TECHNIQUES:
        out[t] = templates[t].format(task=task, examples=format_examples(category, n_examples), role=role)
    return out

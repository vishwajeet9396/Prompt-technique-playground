"""Unit IV: turn one image task into four prompt variants, one per prompting technique.
With an LLM provider the variant is rewritten by the model; otherwise an offline template is used (and labelled so)."""
from config import TECHNIQUES
from modules.llm_provider import generate

FEW_SHOT = ("Examples of good image prompts:\n"
            "- a lighthouse on a cliff at dawn, soft golden light, wide-angle, highly detailed\n"
            "- a bowl of ramen with rising steam, macro photo, shallow depth of field\n")

META = {
    "Few-shot": FEW_SHOT + "\nNow write one image prompt in the same style for: {task}\nOutput ONLY the prompt.",
    "Chain-of-Thought": ("Plan the image by considering subject, setting, lighting, style and composition (do not show your reasoning). "
                         "Then output ONLY the final image prompt as one paragraph under 60 words.\nTask: {task}"),
    "Role-based": ("You are an experienced art director writing prompts for a text-to-image model. "
                   "Write ONE vivid prompt under 60 words for: {task}\nOutput ONLY the prompt."),
}

TEMPLATES = {
    "Zero-shot": "{task}",
    "Few-shot": "{task}, highly detailed, soft golden-hour light, wide-angle composition, sharp focus",
    "Chain-of-Thought": ("Subject: {task}. Setting: a fitting natural environment. Lighting: soft natural light. "
                         "Style: realistic photograph. Composition: centred, rule of thirds. Quality: sharp focus, high detail."),
    "Role-based": "Art director's brief: {task}, cinematic lighting, 35mm lens, professional editorial quality",
}


def make_variants(task: str, provider: str = "Demo (no API)") -> dict:
    """Returns {technique: {original, modified, method}}."""
    out = {}
    use_llm = provider in ("Groq", "Gemini")
    for t in TECHNIQUES:
        modified, method = TEMPLATES[t].format(task=task), "Template (offline)"
        if t == "Zero-shot":
            method = "Task used as written"
        elif use_llm:
            res = generate(META[t].format(task=task), provider, temperature=0.7, max_tokens=150)
            text = res["text"].strip().strip('"').replace("\n", " ")
            if text and not res["error"]:
                modified, method = text[:500], f"LLM rewrite ({provider})"
        out[t] = {"original": task, "modified": modified, "method": method}
    return out

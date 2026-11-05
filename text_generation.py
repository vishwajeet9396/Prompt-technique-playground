"""Unit III: GPT-style text generation lab - same text task through the four techniques."""
from config import TECHNIQUES
from modules.prompt_engine import build_prompts
from modules.llm_provider import generate

TEXT_TASKS = {
    "Summarization": "Summarize in two sentences: Transfer learning reuses a model trained on a large dataset as the starting point for a new, smaller task, which saves time and often improves accuracy when labelled data is limited.",
    "Question answering": "Why does a neural network need a non-linear activation function?",
    "Sentiment analysis": "Classify the sentiment (Positive/Negative/Mixed) and give one reason: 'The delivery was late but the product is excellent.'",
    "Creative writing": "Write a four-line poem about debugging code.",
    "Explanation": "Explain the bias-variance trade-off to a first-year student.",
}


def run_text_task(task: str, category: str, provider: str, temperature: float, max_tokens: int,
                  n_examples: int = 2, role: str = None) -> dict:
    prompts = build_prompts(task, category, n_examples, role)
    return {t: {"prompt": prompts[t], **generate(prompts[t], provider, temperature, max_tokens)} for t in TECHNIQUES}


def to_text_file(task: str, provider: str, results: dict) -> str:
    parts = [f"Task: {task}", f"Provider: {provider}", ""]
    for t in TECHNIQUES:
        parts += [f"=== {t} ===", "Prompt:", results[t]["prompt"], "", "Output:", results[t]["text"] or f"(error: {results[t]['error']})", ""]
    return "\n".join(parts)

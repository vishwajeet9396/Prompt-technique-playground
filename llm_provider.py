"""LLM access: Groq or Gemini (free tiers), plus an honest offline demo mode."""
import time
import importlib.util
from functools import lru_cache
import requests
from config import get_secret, GROQ_MODEL, GEMINI_MODEL


LOCAL = "Local GPT-2 (distilgpt2)"


@lru_cache(maxsize=1)
def _local_pipe():
    from transformers import pipeline
    return pipeline("text-generation", model="distilgpt2")


def available_providers() -> list:
    p = []
    if get_secret("GROQ_API_KEY"):
        p.append("Groq")
    if get_secret("GEMINI_API_KEY"):
        p.append("Gemini")
    if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
        p.append(LOCAL)
    return p + ["Demo (no API)"]


def generate(prompt: str, provider: str, temperature: float = 0.5, max_tokens: int = 400) -> dict:
    """Returns {text, latency, provider, error}. Never raises."""
    start = time.time()
    try:
        if provider == "Groq":
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {get_secret('GROQ_API_KEY')}"},
                json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "max_tokens": max_tokens}, timeout=60)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            r = requests.post(url, params={"key": get_secret("GEMINI_API_KEY")},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}, timeout=60)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif provider == LOCAL:
            pipe = _local_pipe()
            kw = {"do_sample": True, "temperature": max(temperature, 0.1)} if temperature > 0 else {"do_sample": False}
            text = pipe(prompt[-1500:], max_new_tokens=min(max_tokens, 150), return_full_text=False,
                        pad_token_id=50256, **kw)[0]["generated_text"]
        else:
            text = ("[DEMO MODE - not a real model output] No API key is configured, so this is a placeholder. "
                    "Add GROQ_API_KEY or GEMINI_API_KEY to get real generations.\n\nPrompt received:\n" + prompt[:300])
        return {"text": text.strip(), "latency": round(time.time() - start, 2), "provider": provider, "error": None}
    except Exception as e:
        return {"text": "", "latency": round(time.time() - start, 2), "provider": provider, "error": str(e)[:200]}

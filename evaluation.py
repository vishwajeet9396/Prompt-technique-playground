"""Automated scoring: LLM-as-judge (when an API is available) with a transparent heuristic fallback.
Both are APPROXIMATE and may differ from human judgement."""
import json
import re
from config import CRITERIA
from modules.llm_provider import generate


def heuristic_scores(task: str, answer: str) -> dict:
    words = re.findall(r"\w+", answer.lower())
    tw = set(re.findall(r"\w+", task.lower()))
    overlap = len(tw & set(words)) / max(1, len(tw))
    n = len(words)
    sents = max(1, len(re.findall(r"[.!?]", answer)))
    avg_len = n / sents
    clarity = 5 if 8 <= avg_len <= 22 else 3
    completeness = 1 + min(4, n / 40)
    concise = 5 if n <= 120 else max(1, 5 - (n - 120) / 80)
    follow = 4 if answer.strip() else 1
    return {"Relevance": round(1 + 4 * min(1, overlap * 1.5), 1), "Clarity": clarity,
            "Completeness": round(completeness, 1), "Correctness": 3.0,
            "Instruction following": follow, "Conciseness": round(concise, 1)}


def judge(task: str, answer: str, provider: str) -> dict:
    """Return {scores, method}. Falls back to heuristics if the judge call/parse fails."""
    if provider in ("Groq", "Gemini") and answer:
        prompt = (f"Rate the ANSWER to the TASK on each criterion from 1 to 5: {', '.join(CRITERIA)}.\n"
                  f"Reply with ONLY a JSON object mapping each criterion name to a number.\n\nTASK: {task}\n\nANSWER: {answer}")
        res = generate(prompt, provider, temperature=0.0, max_tokens=200)
        try:
            m = re.search(r"\{.*\}", res["text"], re.S)
            data = json.loads(m.group(0))
            scores = {c: float(max(1, min(5, data[c]))) for c in CRITERIA}
            return {"scores": scores, "method": "LLM-as-judge"}
        except Exception:
            pass
    return {"scores": heuristic_scores(task, answer), "method": "Heuristic (approximate)"}


def overall(scores: dict) -> float:
    return round(sum(scores.values()) / len(scores), 2)

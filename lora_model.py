"""Unit II: use a LoRA adapter (trained with train_lora.py) as the fifth contender.
Nothing here claims training happened: it checks for a real adapter on disk."""
import json
import os
from functools import lru_cache
import pandas as pd
from config import TECHNIQUES
from modules.prompt_engine import build_prompts
from modules.llm_provider import generate
from utils.dataset_loader import LABELS

ADAPTER_DIR = "models/lora_adapter"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = "Classify the sentiment of the review as Positive, Negative or Neutral."


def format_prompt(text: str) -> str:
    return f"{INSTRUCTION}\nReview: {text}\nSentiment:"


def parse_label(output: str):
    low = output.lower()
    pos = {l: low.find(l.lower()) for l in LABELS if l.lower() in low}
    return min(pos, key=pos.get) if pos else None


def adapter_exists() -> bool:
    return os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))


def load_training_info():
    p = os.path.join(ADAPTER_DIR, "training_info.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_test_set():
    p = os.path.join(ADAPTER_DIR, "test_set.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = (load_training_info() or {}).get("base_model", BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(base)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), ADAPTER_DIR).eval()
    return tok, model


def predict(text: str) -> dict:
    """Returns {output, label, error}."""
    try:
        import torch
        tok, model = _load()
        ids = tok(format_prompt(text), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {"output": txt, "label": parse_label(txt), "error": None}
    except Exception as e:
        return {"output": "", "label": None, "error": str(e)[:200]}


def compare_on_test_set(test: pd.DataFrame, provider: str, progress=None) -> pd.DataFrame:
    """Run LoRA + four prompt techniques on the held-out test set. Rows: one per (example, contender)."""
    rows, total, done = [], len(test) * 5, 0
    for _, r in test.iterrows():
        task = f"{INSTRUCTION} Reply with one word.\nReview: {r['text']}"
        prompts = build_prompts(task, "Sentiment analysis", 2)
        for t in TECHNIQUES:
            res = generate(prompts[t], provider, temperature=0.0, max_tokens=60)
            lab = parse_label(res["text"]) if not res["error"] else None
            rows.append({"text": r["text"], "truth": r["label"], "contender": t, "predicted": lab, "error": res["error"]})
            done += 1
            progress and progress(done / total)
        p = predict(r["text"])
        rows.append({"text": r["text"], "truth": r["label"], "contender": "LoRA-tuned", "predicted": p["label"], "error": p["error"]})
        done += 1
        progress and progress(done / total)
    return pd.DataFrame(rows)


def accuracy_table(res: pd.DataFrame) -> pd.DataFrame:
    ok = res[res["error"].isna()].copy()
    ok["correct"] = ok["predicted"] == ok["truth"]
    g = ok.groupby("contender").agg(examples=("correct", "count"), correct=("correct", "sum")).reset_index()
    g["accuracy %"] = (100 * g["correct"] / g["examples"]).round(1)
    return g

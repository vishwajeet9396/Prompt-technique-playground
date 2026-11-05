"""Unit V: hallucination experiment across the four prompt techniques.
Method: keyword checks against reference facts. Limitations are shown in the app."""
import re
import pandas as pd
from config import TECHNIQUES
from modules.prompt_engine import build_prompts
from modules.llm_provider import generate

NEG = re.compile(r"(not|n't|rather than|instead of|never|unlike)\W+(\w+\W+){0,3}$")
UNCERTAIN = ["not aware", "cannot verify", "can't verify", "no record", "unable to", "doesn't exist", "does not exist",
             "couldn't find", "could not find", "unfamiliar", "not familiar", "no such", "i don't have", "i do not have",
             "cannot find", "can't find", "not able to", "may not exist", "appears to be fictional", "fictional"]


def _terms(s) -> list:
    return [] if pd.isna(s) else [x.strip().lower() for x in str(s).split("|") if x.strip()]


def _positive_hits(low: str, terms: list) -> list:
    out = []
    for t in terms:
        for m in re.finditer(re.escape(t), low):
            if not NEG.search(low[max(0, m.start() - 30):m.start()]):
                out.append(t)
                break
    return out


def score_answer(row, answer: str) -> dict:
    low = answer.lower()
    if row["kind"] == "trap":
        ok = any(u in low for u in UNCERTAIN)
        return {"correct": ok, "hallucination": not ok,
                "note": "Admitted it cannot verify" if ok else "Invented details about a non-existent paper"}
    key = any(k in low for k in _terms(row["key_facts"]))
    wrong = _positive_hits(low, _terms(row["wrong_claims"]))
    return {"correct": key and not wrong, "hallucination": bool(wrong),
            "note": ("Wrong claim: " + ", ".join(wrong)) if wrong else ("Matches reference" if key else "Missing reference fact")}


def run_experiment(df: pd.DataFrame, provider: str, max_tokens: int = 250, progress=None) -> pd.DataFrame:
    rows, total, done = [], len(df) * len(TECHNIQUES), 0
    for _, q in df.iterrows():
        prompts = build_prompts(q["question"], "Question answering", 2)
        for t in TECHNIQUES:
            res = generate(prompts[t], provider, temperature=0.0, max_tokens=max_tokens)
            if res["error"]:
                s = {"correct": None, "hallucination": None, "note": "API error: " + res["error"]}
            else:
                s = score_answer(q, res["text"])
            rows.append({"id": q["id"], "technique": t, "question": q["question"], "reference": q["reference"],
                         "answer": res["text"], **s})
            done += 1
            if progress:
                progress(done / total)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    ok = res.dropna(subset=["hallucination"])
    g = ok.groupby("technique").agg(questions=("id", "count"), correct=("correct", "sum"),
                                    hallucinations=("hallucination", "sum")).reset_index()
    g["accuracy %"] = (100 * g["correct"] / g["questions"]).round(1)
    g["hallucination rate %"] = (100 * g["hallucinations"] / g["questions"]).round(1)
    return g

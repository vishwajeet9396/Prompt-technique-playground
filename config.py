"""Central configuration. API keys are read from environment / Streamlit secrets, never hard-coded."""
import os

TECHNIQUES = ["Zero-shot", "Few-shot", "Chain-of-Thought", "Role-based"]
CRITERIA = ["Relevance", "Clarity", "Completeness", "Correctness",
            "Instruction following", "Conciseness"]
TASK_CATEGORIES = ["Question answering", "Explanation", "Summarization",
                   "Sentiment analysis", "Creative writing", "Reasoning / maths"]

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def get_secret(name: str):
    """Read a key from env vars first, then Streamlit secrets (if available)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None

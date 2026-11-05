# Deployment Guide - Prompt Technique Playground

Platform limits and screens change often, so check each platform's current docs if a step looks different.

## 0. Choose a requirements file
| File | Contains | Use when |
|---|---|---|
| `requirements-full.txt` | everything incl. torch, transformers, sentence-transformers | Hugging Face Spaces (more RAM), or your own laptop |
| `requirements-light.txt` | no torch/transformers | Streamlit Community Cloud or Render free tier (low RAM) |

Hosting services read a file named `requirements.txt`. Copy the one you want over it before pushing:
```
cp requirements-light.txt requirements.txt      # light deployment
cp requirements-full.txt requirements.txt       # full deployment
```
With the light file, the app still works: BERT similarity uses a labelled TF-IDF fallback, the BERT sentiment classifier, local GPT-2 and the LoRA "try it" box show a clear "unavailable" message, and the hosted LLM (Groq/Gemini) features all work.

## 1. Get free API keys
- Groq: console.groq.com -> API Keys -> create key (`GROQ_API_KEY`)
- Gemini: aistudio.google.com -> Get API key (`GEMINI_API_KEY`)
- Hugging Face (images): huggingface.co/settings/tokens -> new token with read access (`HF_TOKEN`)

You need at least one of Groq or Gemini. Never paste keys into code or commit them.

## 2. Run locally first
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key                         # Windows PowerShell: $env:GROQ_API_KEY="your_key"
streamlit run app.py
```
Open http://localhost:8501 and test every page before deploying.

## 3. Create the GitHub repository
1. github.com -> New repository -> name `prompt-technique-playground` -> Public (or Private) -> Create.
2. In the project folder:
```
git init
git add .
git commit -m "Prompt Technique Playground"
git branch -M main
git remote add origin https://github.com/<your-username>/prompt-technique-playground.git
git push -u origin main
```
3. Check on GitHub that `.env` and `.streamlit/secrets.toml` are NOT present. `.gitignore` already excludes them.
4. If you trained LoRA, make sure `models/lora_adapter/` (a few MB) is committed.

## 4A. Streamlit Community Cloud (easiest)
1. Go to share.streamlit.io and sign in with GitHub.
2. New app -> choose your repo, branch `main`, main file `app.py`.
3. Advanced settings -> Secrets, paste TOML:
```
GROQ_API_KEY = "your_key"
GEMINI_API_KEY = "your_key"
HF_TOKEN = "your_token"
```
4. Deploy. The first build takes a few minutes. Use `requirements-light.txt` as `requirements.txt` because RAM is limited.

## 4B. Hugging Face Spaces (best for the full version)
1. huggingface.co -> New Space -> choose the Streamlit SDK (if it is not offered, choose Docker and its Streamlit template) -> CPU Basic (free).
2. Put this block at the very top of `README.md` in the Space repo:
```
---
title: Prompt Technique Playground
sdk: streamlit
app_file: app.py
---
```
3. Push your code to the Space repo (`git remote add space https://huggingface.co/spaces/<user>/<space>` then `git push space main`), or upload files in the web UI.
4. Space -> Settings -> Variables and secrets -> add `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN` as **secrets**.
5. The Space rebuilds automatically. Use `requirements-full.txt` if you want local BERT models; the first run downloads the models.

## 4C. Render (alternative)
1. render.com -> New -> Web Service -> connect the GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Environment -> add the three keys as environment variables.
5. Use the light requirements. Free services sleep when idle, so the first request after a pause is slow.

## 5. Test the public URL
- [ ] Home loads and the sidebar shows your provider (not only "Demo").
- [ ] Playground: run one task, four outputs appear.
- [ ] Rating & Evaluation: save a run and score it.
- [ ] Hallucination and Injection pages run (they need a real provider).
- [ ] Image page: variants appear; images appear if `HF_TOKEN` works.
- [ ] Report Generator: PDF and Word download.
- [ ] No keys visible anywhere in the page or logs.

## 6. Troubleshooting
| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` | Package missing from `requirements.txt`; add it and redeploy |
| Sidebar shows only "Demo (no API)" | Secret name wrong or not saved. Names must be exactly `GROQ_API_KEY` / `GEMINI_API_KEY`; restart the app |
| API error 401 | Key invalid or expired; create a new one |
| API error 429 | Free-tier rate limit; wait a minute, reduce experiments |
| Gemini/Groq model not found | Model names change; set `GROQ_MODEL` / `GEMINI_MODEL` env vars to a current model |
| Image error HTTP 404/410/503 | Hosted endpoint changed or model is loading; set `HF_IMAGE_URL` to the current endpoint from Hugging Face docs, retry later |
| App crashes or restarts at start | Out of memory; switch to `requirements-light.txt` |
| Build very slow or fails on torch | Use the light file, or deploy on Hugging Face Spaces |
| LoRA page says no adapter | `models/lora_adapter/` not committed; train, copy the folder, push again |
| Results vanished | Results live only in the session; download the ZIP/CSV before closing |

"""Unit IV: text-to-image backends. Never returns a made-up image: if a backend fails you get an error message."""
import importlib.util
import io
import time
from functools import lru_cache
import requests
from config import get_secret

HF_BACKEND = "Hugging Face Inference API"
LOCAL_BACKEND = "Local Diffusers (sd-turbo, CPU/GPU)"
NONE_BACKEND = "None (prompts only)"
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# Hosted endpoints change over time; override with the HF_IMAGE_URL environment variable if this one stops working.
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


def available_backends() -> list:
    b = []
    if get_secret("HF_TOKEN"):
        b.append(HF_BACKEND)
    if importlib.util.find_spec("diffusers") and importlib.util.find_spec("torch"):
        b.append(LOCAL_BACKEND)
    return b + [NONE_BACKEND]


def _to_png(raw: bytes) -> bytes:
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    img.load()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=1)
def _local_pipe():
    import torch
    from diffusers import AutoPipelineForText2Image
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(prompt: str, backend: str, seed: int = 42) -> dict:
    """Returns {png, error, latency, backend}. png is None on failure."""
    start = time.time()
    try:
        if backend == HF_BACKEND:
            import os
            r = requests.post(os.getenv("HF_IMAGE_URL", HF_URL), headers={"Authorization": f"Bearer {get_secret('HF_TOKEN')}"},
                              json={"inputs": prompt, "parameters": {"seed": seed}}, timeout=120)
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
            png = _to_png(r.content)
        elif backend == LOCAL_BACKEND:
            import torch
            pipe = _local_pipe()
            gen = torch.Generator().manual_seed(seed)
            img = pipe(prompt, num_inference_steps=2, guidance_scale=0.0, generator=gen).images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
        else:
            raise RuntimeError("No image backend selected, so no image was generated.")
        return {"png": png, "error": None, "latency": round(time.time() - start, 1), "backend": backend}
    except Exception as e:
        return {"png": None, "error": str(e)[:200], "latency": round(time.time() - start, 1), "backend": backend}

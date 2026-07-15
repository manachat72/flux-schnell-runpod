"""
RunPod Serverless handler — FLUX.1 [schnell]  (Apache-2.0, commercial OK)

Input (job["input"]):
  prompt   : str   (required)
  width    : int   (default 1024)
  height   : int   (default 1024)
  steps    : int   (default 4)   schnell best at 1-4
  seed     : int   (optional)
  guidance : float (default 0.0) schnell uses 0
Output:
  { "image_base64": "<PNG b64>", "seed": int, "width": w, "height": h }
"""
import base64, io, os, random
import torch, runpod
from diffusers import FluxPipeline

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-schnell")

print(f"[init] loading {MODEL_ID} ...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
pipe.enable_attention_slicing()
print("[init] model ready")

MAX_SEED = 2**32 - 1

def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "missing 'prompt'"}
    width  = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps  = int(inp.get("steps", 4))
    guidance = float(inp.get("guidance", 0.0))
    seed = inp.get("seed")
    seed = random.randint(0, MAX_SEED) if seed is None else int(seed) % (MAX_SEED + 1)
    gen = torch.Generator("cuda").manual_seed(seed)
    try:
        out = pipe(prompt=str(prompt), width=width, height=height,
                   num_inference_steps=steps, guidance_scale=guidance,
                   generator=gen, max_sequence_length=256)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "CUDA OOM - reduce size or bigger GPU"}
    buf = io.BytesIO(); out.images[0].save(buf, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode(),
            "seed": seed, "width": width, "height": height, "steps": steps}

runpod.serverless.start({"handler": handler})

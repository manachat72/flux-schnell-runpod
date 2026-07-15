# FLUX.1 [schnell] RunPod Serverless worker
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake weights into the image so cold starts don't re-download ~24GB.
# Needs a HF token at build time for the gated repo:  --build-arg HF_TOKEN=hf_xxx
ARG HF_TOKEN
ENV HF_HOME=/app/hf
RUN python -c "import os; from huggingface_hub import login; \
    tok=os.environ.get('HF_TOKEN'); login(tok) if tok else None" ; \
    python -c "import torch; from diffusers import FluxPipeline; \
    FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.bfloat16)"

COPY handler.py .
CMD ["python", "-u", "handler.py"]

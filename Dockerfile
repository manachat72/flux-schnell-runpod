# FLUX.1 [schnell] RunPod Serverless worker
# โมเดลโหลดตอน runtime ผ่าน Cached model ของ RunPod (+ HF token ที่ตั้งใน endpoint)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# base image มี flash-attn 3 ที่ register torch custom-op ด้วย schema ที่ torch infer ไม่ได้
# ทำให้ import diffusers พังตอน load — ถอดออก diffusers จะใช้ attention ปกติแทน
RUN pip uninstall -y flash-attn flash_attn transformer-engine transformer_engine || true

COPY handler.py .
CMD ["python", "-u", "handler.py"]

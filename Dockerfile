# FLUX.1 [schnell] RunPod Serverless worker
# โมเดลโหลดตอน runtime ผ่านฟีเจอร์ Cached model ของ RunPod (+ HF token ที่ตั้งใน endpoint)
# ไม่ bake weights ตอน build เพื่อเลี่ยงปัญหา gated auth ระหว่าง build
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]

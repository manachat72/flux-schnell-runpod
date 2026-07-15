# FLUX.1 [schnell] RunPod Serverless worker
# ใช้ PyTorch official image (สะอาด ไม่มี flash-attn ที่ทำให้ diffusers import พัง)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]

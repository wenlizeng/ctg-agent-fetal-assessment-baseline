FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ctg_baseline ./ctg_baseline
COPY checkpoints ./checkpoints
COPY models ./models

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "-m", "ctg_baseline.run"]

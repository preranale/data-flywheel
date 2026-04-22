# ============================================================
# Trainer Dockerfile
# ============================================================
# This container runs ONCE per training job then exits.
# It is triggered by the scheduler, not kept alive.
#
# WHY a separate container for training?
# Training is resource-intensive. Running it in a separate
# container means:
# - It can be given more CPU/memory than other services
# - In production it could run on a GPU instance
# - It doesn't compete with the API for resources
# ============================================================

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No EXPOSE, no long-running server
# Just runs train.py and exits
CMD ["python", "train.py"]

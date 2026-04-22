# ============================================================
# Scheduler Dockerfile
# Needs Docker CLI inside the container so it can trigger
# "docker compose run trainer" to start a training job.
# We use docker:dind (Docker-in-Docker) base for this.
# ============================================================

FROM python:3.11-slim

# Install Docker CLI (not daemon — just the client)
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scheduler.py"]

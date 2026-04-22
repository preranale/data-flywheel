# ============================================================
# Feature Pipeline Dockerfile
# This is a worker — no HTTP server, no exposed port.
# It just runs pipeline.py in a loop forever.
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for downloading/unzipping dataset
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No EXPOSE — this service has no HTTP interface
CMD ["python", "pipeline.py"]

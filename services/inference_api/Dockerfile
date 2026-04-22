# ============================================================
# Inference API Dockerfile
# ============================================================
# We use python:3.11-slim (not full python:3.11) because:
# - "slim" strips out compilers, docs, and test files
# - Result: ~150MB image instead of ~900MB
# - Faster to build, faster to pull, less attack surface
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — before copying source code.
# WHY: Docker caches layers. If we copy source first,
# every code change invalidates the pip install cache.
# This order means pip only re-runs when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual source code
COPY . .

# Expose port 8000 so docker-compose can map it
EXPOSE 8000

# Start the server
# --host 0.0.0.0 means "accept connections from outside the container"
# --reload means "restart when code changes" (dev mode only)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

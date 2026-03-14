# =============================================================================
# PdM Agent - Multi-stage Dockerfile
# =============================================================================

# --- Builder stage ---
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --user --no-cache-dir --upgrade -r /build/requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --user --no-cache-dir opencv-python-headless

# --- Production stage ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/workspace/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

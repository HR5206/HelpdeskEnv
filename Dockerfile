FROM python:3.11-slim

WORKDIR /app

# Make pip a bit quieter and avoid cache bloat
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/.cache/huggingface
ENV TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers

# Install runtime dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p /data/.cache/huggingface/transformers

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip build-essential curl tesseract-ocr tesseract-ocr-eng \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Pre-compiled CUDA wheel (Works on Linux, fails on WSL2 Windows)
RUN python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY . .
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN useradd -m -u 1000 nyaya && chown -R nyaya:nyaya /app
USER nyaya

EXPOSE 8501 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
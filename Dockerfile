# Dockerfile for Hugging Face Spaces (free CPU tier)
# ----------------------------------------------------
# HF Spaces expects:
#   - App listens on 0.0.0.0:7860
#   - Runs as a non-root user that can write to $HOME
#
# For a free CPU Space, we use CPU-only torch to keep the image small (< 10 GB).
# To run on GPU later, switch the torch install line to the matching CUDA wheel.

FROM python:3.11-slim

# System deps: ffmpeg for H.264 re-encoding; libgl/libglib for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# HF Spaces wants a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR $HOME/app

# Install CPU-only torch first (smaller image, faster build on free tier)
RUN pip install --user --no-cache-dir \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Then the rest of the requirements
COPY --chown=user:user backend/requirements.txt ./requirements.txt
# Remove torch/torchvision from requirements since they're already installed above
RUN grep -v -E '^torch(vision)?' requirements.txt > reqs-rest.txt && \
    pip install --user --no-cache-dir -r reqs-rest.txt

# Copy the project
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user frontend/ ./frontend/

# Storage needs to be writable
RUN mkdir -p ./backend/storage ./backend/models

# HF Spaces runs on port 7860
ENV PORT=7860

# IMPORTANT: you must add your .pt checkpoint into backend/models/best.pt
# before building, OR upload it as a Space secret + download at startup.
# See README.md "Deploying to Hugging Face Spaces".

WORKDIR $HOME/app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

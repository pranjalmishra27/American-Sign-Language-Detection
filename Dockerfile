# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — ASL Detection System
#
# Build (CPU):
#   docker build -t asl-detection .
#
# Build (GPU — requires nvidia-docker):
#   docker build --build-arg BASE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
#                -t asl-detection-gpu .
#
# Run webcam demo (requires X11 forwarding on Linux):
#   docker run --rm -it \
#     --device /dev/video0 \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/outputs:/app/outputs \
#     asl-detection python src/webcam.py
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE=python:3.10-slim-bullseye
FROM ${BASE}

# ── system deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── project code ──────────────────────────────────────────────────────────────
COPY . .

# ── create output dirs ────────────────────────────────────────────────────────
RUN mkdir -p models outputs dataset/images/{train,valid,test} \
             dataset/labels/{train,valid,test}

# ── environment defaults (overridden by .env or docker-compose) ───────────────
ENV MODEL_PATH=models/asl_yolov8n/weights/best.pt \
    CONFIDENCE_THRESHOLD=0.5 \
    CAMERA_INDEX=0 \
    PYTHONUNBUFFERED=1

# ── default command: show help ────────────────────────────────────────────────
CMD ["python", "src/webcam.py", "--help"]

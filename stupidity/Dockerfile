FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system packages for common scientific Python wheels and SSL certificates.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libportmidi0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to maximize layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the project sources.
COPY . /app

# Default to the voxelnet pipeline.
ENTRYPOINT ["python", "-m", "spatial_intelligence.threat_modelling.voxelnet_pipeline"]
CMD ["--mode", "sim", "--steps", "300", "--record-dir", "/app/visualizations/frames"]

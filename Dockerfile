# NS-Gym ICCPS hands-on tutorial — base image.
#
# Build:  docker compose build
# Run:    docker compose up
# Open:   http://localhost:8888  (auth disabled for local tutorial use)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# Bring in uv (Astral's fast package resolver) from the official image. uv's
# PubGrub resolver avoids the pip backtracking that lands on legacy gym==0.21.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Pull current Debian security patches, then install the system libraries
# MuJoCo + matplotlib dlopen at runtime (headless rendering).
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglu1-mesa \
        libosmesa6 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first. The default wheel pulls CUDA libraries (~2.5 GB
# of libtorch_cuda.so etc.) which the container can't use anyway — Docker
# containers don't expose host GPUs to MuJoCo/Torch in this setup.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# ns-gym pulls in gymnasium[mujoco], mujoco, stable-baselines3[extra], matplotlib,
# pandas, etc. transitively. Marimo is the reactive notebook server — its sliders
# and dataflow re-execution showcase parametric non-stationarity better than
# Jupyter's imperative cells. uv will see torch is already satisfied.
RUN uv pip install ns-gym marimo

WORKDIR /workspace

COPY nb_*.py ./
COPY README.md ./

EXPOSE 8888

CMD ["marimo", "edit", \
     "--host", "0.0.0.0", \
     "--port", "8888", \
     "--headless", \
     "--no-token", \
     "/workspace"]

     
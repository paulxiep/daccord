# CPU dev image for d'accord — shared by root / eval / audit / consumer / serving
# compose services.
#
# Why python:3.14-slim-bookworm and not a CUDA image: most services don't need
# GPU; smaller image = faster pull + smaller host disk footprint. Bakeoff and
# (future) training use Dockerfile.cuda instead.
#
# Why pip install uv: uv has no official conda channel and no Debian package;
# pip into the system Python is the supported install path. This is the ONE
# pip install we accept — everything else routes through uv.
#
# This Dockerfile is intentionally minimal. Production reuse (ECR for
# SageMaker / AgentCore in Phase 2) should derive from this via multi-stage
# builds — keep the base lean.

FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    DEBIAN_FRONTEND=noninteractive

# git: needed by uv when a dep is sourced from a git URL.
# build-essential: only used as a fallback when uv can't find a wheel; on
# Linux + 3.14 this should be rare (much better wheel coverage than Windows).
# ca-certificates: HTTPS to PyPI / huggingface.co.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# No COPY of project source — compose bind-mounts /workspace at run time.
# That keeps image rebuilds independent of code changes (only Dockerfile +
# system deps trigger rebuild).

# VICTOR Container Image
# Enterprise-grade autonomous intelligence system

# Build stage for dependencies
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Production stage
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS production

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels following OCI image spec
LABEL org.opencontainers.image.title="VICTOR" \
      org.opencontainers.image.description="Enterprise-grade Autonomous Intelligence System" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.vendor="MASSIVEMAGNETICS" \
      org.opencontainers.image.licenses="Proprietary"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Victor-specific environment
    DEPLOYMENT_MODE=prod \
    LOG_LEVEL=INFO \
    CHECKPOINT_DIR=/data/checkpoints \
    MODEL_DIR=/data/models

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash victor

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory and data directories
WORKDIR /app
RUN mkdir -p /data/checkpoints /data/models && \
    chown -R victor:victor /app /data

# Copy application code
COPY --chown=victor:victor victor_infrastructure.py .
COPY --chown=victor:victor victor_integration.py .

# Create health check script
RUN echo '#!/bin/bash\ncurl -sf http://localhost:8000/health/ready || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Create shutdown script
RUN echo '#!/bin/bash\necho "Graceful shutdown initiated"\nkill -SIGTERM 1' > /app/shutdown.sh && \
    chmod +x /app/shutdown.sh

# Switch to non-root user
USER victor

# Expose ports
# 8000 - HTTP API
# 8001 - gRPC API  
# 9090 - Prometheus metrics
EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Default command
CMD ["python", "/app/victor_integration.py", "--mode", "prod"]

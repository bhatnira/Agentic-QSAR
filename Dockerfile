# CTA-QSAR — CPU-first container.
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# System deps for RDKit (as wheels) and light ML runtimes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install -e .

# Optional extras (uncomment for GPU or ML extras):
# RUN pip install ".[gpu,ml]"

COPY .env.example .env.example

RUN mkdir -p /app/runs /app/data
VOLUME ["/app/runs", "/app/data"]

ENTRYPOINT ["cta-qsar"]
CMD ["--help"]

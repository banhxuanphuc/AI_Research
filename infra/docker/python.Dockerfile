# Image dùng chung cho api + worker (+ chạy test/eval).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

# CA bổ sung (vd antivirus TLS scanning) - xem infra/docker/certs/README.md
COPY infra/docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_SYSTEM_CERTS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Cài dependency trước (cache layer), code sau
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "research_scientist.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

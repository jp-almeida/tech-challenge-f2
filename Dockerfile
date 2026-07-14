FROM python:3.10-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
RUN pip install poetry==1.8.3 \
    && poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root --no-interaction --no-ansi

FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY dvc.yaml ./
COPY .dvc ./.dvc

CMD ["python", "-m", "src.training.train"]

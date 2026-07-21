FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY scimage/ ./scimage/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[dev,anndata]"

CMD ["pytest", "--tb=short"]

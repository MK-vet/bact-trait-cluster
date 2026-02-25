FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[all]" 2>/dev/null || pip install --no-cache-dir .
COPY examples/ examples/
CMD ["bact-trait-cluster", "--help"]

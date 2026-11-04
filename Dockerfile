# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY harness harness
COPY bench bench
COPY triage_bridge triage_bridge
COPY evolve evolve
RUN pip install --no-cache-dir --prefix=/install ".[mcp,llm]"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /install /usr/local
RUN mkdir -p /app/workspace /app/skills /app/data && chown -R app:app /app
USER app
ENV HARNESS_WORKSPACE=/app/workspace HARNESS_SKILLS=/app/skills HARNESS_DB=/app/data/runs.db
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "harness.api:app", "--host", "0.0.0.0", "--port", "8000"]

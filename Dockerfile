FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/marketops
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
COPY api ./api
COPY app ./app
RUN python -m pip install --no-cache-dir -c constraints.txt ".[train]" \
    && groupadd --system marketops \
    && useradd --system --gid marketops --home-dir /tmp marketops

USER marketops

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
USER root
RUN python -m pip install --no-cache-dir -c constraints.txt ".[ui]"
USER marketops
ENV HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

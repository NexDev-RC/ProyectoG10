# ──────────────────────────────────────────────────────────────
#  Olist Revenue Forecast – Imagen base compartida (API + Dashboard)
#  Sprint 4 – Integración y despliegue
#
#  El mismo Dockerfile sirve a los servicios `api` y `dashboard`;
#  docker-compose.yml sobreescribe el `command` de cada uno.
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Evita prompts de apt y archivos .pyc; logs sin buffer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias de sistema mínimas (LightGBM requiere libgomp1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Instala dependencias Python primero (capa cacheable)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el código y los artefactos necesarios en runtime
COPY src/       ./src/
COPY api/       ./api/
COPY dashboard/ ./dashboard/
COPY config/    ./config/
COPY data/models/    ./data/models/
COPY data/processed/ ./data/processed/
COPY ./data /app/data
#COPY ./mlruns /app/mlruns
RUN mkdir -p /app/mlruns

# Puertos: API (8000), Dashboard (8501)
EXPOSE 8000 8501

# Comando por defecto: API (el dashboard lo sobreescribe en compose)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

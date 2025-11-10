FROM python:3.11.9-slim

WORKDIR /app

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o projeto
COPY . /app

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501

# Rodar FastAPI e Streamlit juntos
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]

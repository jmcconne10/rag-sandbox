FROM python:3.11-slim

# 1) System deps needed to build PyStemmer (and a few others)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Install Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 3) Copy app code
COPY . .

# 4) Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

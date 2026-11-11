FROM python:3.15.2-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY web ./web
COPY models ./models
COPY data/SOLUSDT_15m.csv ./data/
ENV PORT=8000
EXPOSE 8000
CMD uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT}

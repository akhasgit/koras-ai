FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py config.py ./
COPY routes/ routes/
COPY controllers/ controllers/
COPY services/ services/
COPY models/ models/
COPY prompts/ prompts/
COPY utils/ utils/

ENV PORT=8000

EXPOSE $PORT

CMD gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    --timeout 180 \
    --bind 0.0.0.0:$PORT \
    --access-logfile -

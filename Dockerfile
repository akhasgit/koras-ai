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

EXPOSE 8000

# Workers = (2 × cores) + 1 is standard; start with 4 for a 2-core container.
# --timeout 180 covers worst-case: Whisper (5s) + 2x Claude (8s each) + overhead.
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--timeout", "180", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-"]

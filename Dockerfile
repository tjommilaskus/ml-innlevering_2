# The website, ready for any host that runs containers (Render, Fly.io, Hugging Face Spaces, ...)
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# btc_1h.csv is copied in too, so the site starts with all prices until the last commit and only fetches newer hours
COPY app.py btc_data.py btc_model.py btc_1h.csv ./
COPY templates templates
COPY static static

# The host sets PORT. One process keeps memory low; threads let it answer several visitors at once
ENV PORT=8000
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 app:app

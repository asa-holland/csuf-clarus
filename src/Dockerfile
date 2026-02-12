FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY clarus/ clarus/
COPY app.py .
COPY clarus/templates/ templates/
COPY clarus/static/ static/

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]

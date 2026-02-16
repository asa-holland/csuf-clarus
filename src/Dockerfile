FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('The system shall process data.'); assert len(doc) > 0; assert hasattr(doc[0], 'pos_'); assert hasattr(doc[0], 'dep_'); print('✅ SpaCy functionality validated')"

COPY clarus/ clarus/
COPY app.py .
COPY clarus/templates/ templates/
COPY clarus/static/ static/

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]

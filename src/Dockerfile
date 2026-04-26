FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/root/nltk_data'); nltk.download('stopwords', download_dir='/root/nltk_data'); nltk.download('wordnet', download_dir='/root/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/root/nltk_data')"

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

RUN python -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('The system shall process data.'); assert len(doc) > 0; assert hasattr(doc[0], 'pos_'); assert hasattr(doc[0], 'dep_'); print('✅ SpaCy functionality validated')"

# Pre-download ML models to cache (requires internet during build).
# Failures are non-fatal: the image still builds and models are fetched at runtime.
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/transformers
COPY download_models.py .
RUN python download_models.py || echo "Warning: one or more models could not be pre-downloaded"

# Create cache directories and set permissions
RUN mkdir -p /root/.cache/huggingface /root/.cache/transformers && chmod -R 755 /root/.cache

COPY clarus/ clarus/
COPY app.py .
COPY clarus/templates/ templates/
COPY clarus/static/ static/

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]

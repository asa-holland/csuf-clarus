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

# Pre-download transformer models to cache (requires internet during build)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/transformers
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base', cache_dir='/root/.cache/transformers'); print('RoBERTa tokenizer downloaded')" \
        || echo "Warning: RoBERTa tokenizer download failed, will download at runtime" ; \
    python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('roberta-base', cache_dir='/root/.cache/transformers'); print('RoBERTa sequence classification model downloaded')" \
        || echo "Warning: RoBERTa sequence classification download failed" ; \
    python -c "from transformers import AutoModelForTokenClassification; AutoModelForTokenClassification.from_pretrained('roberta-base', cache_dir='/root/.cache/transformers'); print('RoBERTa token classification model downloaded')" \
        || echo "Warning: RoBERTa token classification download failed" ; \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/root/.cache/transformers'); print('Sentence transformer model downloaded')" \
        || echo "Warning: Sentence transformer download failed"

# Create cache directories and set permissions
RUN mkdir -p /root/.cache/huggingface /root/.cache/transformers && chmod -R 755 /root/.cache

COPY clarus/ clarus/
COPY app.py .
COPY clarus/templates/ templates/
COPY clarus/static/ static/

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]

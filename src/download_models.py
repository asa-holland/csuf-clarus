"""Pre-download ML models during Docker build.

Running this script at build time caches all models inside the image so the
application starts instantly without any network access at runtime.
"""
import os
import sys

os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/transformers"


def download_nli_model():
    from sentence_transformers import CrossEncoder

    model_name = "cross-encoder/nli-deberta-v3-base"
    print(f"Downloading {model_name} …", flush=True)
    CrossEncoder(model_name)
    print(f"{model_name} — OK", flush=True)


def download_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading {model_name} …", flush=True)
    SentenceTransformer(model_name, cache_folder="/root/.cache/transformers")
    print(f"{model_name} — OK", flush=True)


def main():
    errors = []

    for fn in (download_nli_model, download_sentence_transformer):
        try:
            fn()
        except Exception as exc:
            msg = f"WARNING: {fn.__name__} failed: {exc}"
            print(msg, flush=True)
            errors.append(msg)

    if errors:
        print("\nSome models could not be pre-downloaded. They will be fetched at runtime.", flush=True)
        sys.exit(1)
    else:
        print("\nAll models pre-downloaded successfully.", flush=True)


if __name__ == "__main__":
    main()

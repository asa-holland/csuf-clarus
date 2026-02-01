import json
import zipfile
from pathlib import Path
from .preprocess import extract_text_from_file, allowed_file


def create_corpus_zip(corpus_dir_path, zip_path):
    corpus_dir = Path(corpus_dir_path)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in corpus_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(corpus_dir)
                zipf.write(file_path, arcname)

    return zip_path


def preprocess_corpus_files(corpus_dir_path, output_dir_path):
    corpus_dir = Path(corpus_dir_path)
    output_dir = Path(output_dir_path)

    output_dir.mkdir(exist_ok=True)

    files_processed = 0
    files_skipped = 0
    errors = []

    for file_path in corpus_dir.iterdir():
        if not file_path.is_file():
            continue

        filename = file_path.name

        if not allowed_file(filename):
            files_skipped += 1
            continue

        try:
            file_extension = filename.rsplit(".", 1)[1].lower()
            text, metadata = extract_text_from_file(str(file_path), file_extension)

            base_name = file_path.stem

            content_file = output_dir / f"{base_name}_content.txt"
            with open(content_file, "w", encoding="utf-8") as f:
                f.write(text)

            metadata_file = output_dir / f"{base_name}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            files_processed += 1

        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")
            files_skipped += 1

    return {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "errors": errors,
    }


def get_corpus_stats(corpus_dir_path):
    corpus_dir = Path(corpus_dir_path)

    if not corpus_dir.exists():
        return None

    total_files = 0
    supported_files = 0
    file_types = {}

    for file_path in corpus_dir.iterdir():
        if not file_path.is_file():
            continue

        total_files += 1

        if allowed_file(file_path.name):
            supported_files += 1

            ext = file_path.suffix.lower().lstrip(".")
            file_types[ext] = file_types.get(ext, 0) + 1

    return {
        "total_files": total_files,
        "supported_files": supported_files,
        "file_types": file_types,
    }

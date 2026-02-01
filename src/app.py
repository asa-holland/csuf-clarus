from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from clarus.steps.preprocess import extract_text_from_file, allowed_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "File type not supported. Allowed types: PDF, DOCX, HTML, TXT"
                    }
                ),
                400,
            )

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        file_extension = filename.rsplit(".", 1)[1].lower()
        text, metadata = extract_text_from_file(file_path, file_extension)

        metadata.update(
            {
                "original_filename": file.filename,
                "file_size": os.path.getsize(file_path),
                "processed_at": datetime.now().isoformat(),
                "text_length": len(text),
            }
        )

        os.remove(file_path)

        return jsonify({"success": True, "text": text, "metadata": metadata})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

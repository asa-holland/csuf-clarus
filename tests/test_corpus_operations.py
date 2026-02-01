import pytest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from clarus.steps.corpus_operations import (
    create_corpus_zip,
    preprocess_corpus_files,
    get_corpus_stats,
)


class TestCreateCorpusZip:

    @pytest.mark.parametrize(
        "file_structure,expected_files",
        [
            (["file1.txt", "file2.txt"], ["file1.txt", "file2.txt"]),
            (
                ["file1.txt", "subdir/file2.txt", "subdir/nested/file3.txt"],
                ["file1.txt", "subdir/file2.txt", "subdir/nested/file3.txt"],
            ),
            ([], []),
            (["single.txt"], ["single.txt"]),
        ],
    )
    def test_should_create_zip_with_various_structures(
        self, file_structure, expected_files
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()

            for file_path in file_structure:
                full_path = corpus_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(f"Content of {file_path}")

            zip_path = Path(temp_dir) / "test.zip"
            create_corpus_zip(corpus_dir, zip_path)

            assert zip_path.exists()
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = zf.namelist()
                assert len(file_list) == len(expected_files)
                for expected_file in expected_files:
                    assert expected_file in file_list

    @pytest.mark.parametrize(
        "nonexistent_path",
        [
            "nonexistent",
            "path/to/nonexistent",
            "/absolute/nonexistent/path",
        ],
    )
    def test_should_raise_error_for_nonexistent_directories(self, nonexistent_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / nonexistent_path
            zip_path = Path(temp_dir) / "test.zip"

            with pytest.raises(FileNotFoundError, match="Corpus directory not found"):
                create_corpus_zip(nonexistent_dir, zip_path)


class TestPreprocessCorpusFiles:

    @pytest.mark.parametrize(
        "file_types,expected_processed,expected_skipped",
        [
            (["file1.txt", "file2.txt"], 2, 0),
            (["file1.jpg", "file2.mp3"], 0, 2),  # All unsupported
            (["file1.txt", "file2.jpg", "file3.pdf"], 2, 1),  # Mixed
            ([], 0, 0),  # Empty
            (["single.txt"], 1, 0),  # Single file
        ],
    )
    @patch("clarus.steps.corpus_operations.allowed_file")
    @patch("clarus.steps.corpus_operations.extract_text_from_file")
    def test_should_process_various_file_combinations(
        self,
        mock_extract,
        mock_allowed,
        file_types,
        expected_processed,
        expected_skipped,
    ):
        def mock_allowed_func(filename):
            return filename.endswith((".txt", ".pdf"))

        mock_allowed.side_effect = mock_allowed_func
        mock_extract.return_value = (
            "Extracted text",
            {"format": "TXT", "author": "Test"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()

            for filename in file_types:
                (corpus_dir / filename).write_text("Content")

            output_dir = Path(temp_dir) / "output"
            result = preprocess_corpus_files(corpus_dir, output_dir)

            assert result["files_processed"] == expected_processed
            assert result["files_skipped"] == expected_skipped
            assert len(result["errors"]) == 0

    @pytest.mark.parametrize(
        "error_scenario,expected_errors",
        [
            ("extraction_error", 3),
            ("partial_errors", 2),
            ("all_errors", 3),
        ],
    )
    @patch("clarus.steps.corpus_operations.allowed_file")
    @patch("clarus.steps.corpus_operations.extract_text_from_file")
    def test_should_handle_various_error_scenarios(
        self, mock_extract, mock_allowed, error_scenario, expected_errors
    ):
        mock_allowed.return_value = True

        def mock_extract_error(file_path, file_extension):
            if error_scenario == "extraction_error":
                raise Exception("Extraction failed")
            elif error_scenario == "partial_errors":
                if "file2" in file_path or "file3" in file_path:
                    raise Exception("Partial error")
                return ("Success", {"format": "TXT"})
            elif error_scenario == "all_errors":
                raise Exception("All failed")

        mock_extract.side_effect = mock_extract_error

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()

            for i in range(3):
                (corpus_dir / f"file{i+1}.txt").write_text("Content")

            output_dir = Path(temp_dir) / "output"
            result = preprocess_corpus_files(corpus_dir, output_dir)

            assert len(result["errors"]) == expected_errors
            if error_scenario == "partial_errors":
                assert result["files_processed"] == 1
                assert result["files_skipped"] == 2


class TestGetCorpusStats:

    @pytest.mark.parametrize(
        "files,expected_total,expected_supported,expected_types",
        [
            (
                ["file1.txt", "file2.pdf", "file3.docx", "file4.jpg", "file5.html"],
                5,
                4,
                {"txt": 1, "pdf": 1, "docx": 1, "html": 1},
            ),
            (["document1.txt", "document2.txt", "document3.txt"], 3, 3, {"txt": 3}),
            (["image1.jpg", "image2.png", "image3.gif"], 3, 0, {}),
            ([], 0, 0, {}),
            (
                ["mixed.txt", "unsupported.mp3", "another.pdf"],
                3,
                2,
                {"txt": 1, "pdf": 1},
            ),
        ],
    )
    def test_should_calculate_stats_for_various_file_sets(
        self, files, expected_total, expected_supported, expected_types
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()

            for filename in files:
                if filename.endswith(".txt"):
                    (corpus_dir / filename).write_text("Text content")
                else:
                    (corpus_dir / filename).write_bytes(b"Binary content")

            stats = get_corpus_stats(corpus_dir)

            assert stats["total_files"] == expected_total
            assert stats["supported_files"] == expected_supported
            assert stats["file_types"] == expected_types

    @pytest.mark.parametrize(
        "directory_setup,should_exist",
        [
            (lambda d: None, False),  # Nonexistent directory
            (lambda d: (Path(d) / "empty").mkdir(), True),  # Empty directory
        ],
    )
    def test_should_handle_directory_scenarios(self, directory_setup, should_exist):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"

            if should_exist:
                corpus_dir.mkdir()

            stats = get_corpus_stats(corpus_dir)

            if should_exist:
                assert stats is not None
                assert stats["total_files"] == 0
                assert stats["supported_files"] == 0
                assert stats["file_types"] == {}
            else:
                assert stats is None


class TestIntegration:

    def test_should_process_real_corpus_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            corpus_dir.mkdir()

            (corpus_dir / "document1.txt").write_text(
                "This is document 1.\nWith multiple lines."
            )
            (corpus_dir / "document2.txt").write_text("This is document 2.")
            (corpus_dir / "image.jpg").write_bytes(b"fake image data")

            stats = get_corpus_stats(corpus_dir)
            assert stats["total_files"] == 3
            assert stats["supported_files"] == 2

            output_dir = Path(temp_dir) / "output"
            result = preprocess_corpus_files(corpus_dir, output_dir)

            assert result["files_processed"] == 2
            assert result["files_skipped"] == 1

            zip_path = Path(temp_dir) / "processed.zip"
            create_corpus_zip(output_dir, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = zf.namelist()
                assert "document1_content.txt" in file_list
                assert "document1_metadata.json" in file_list
                assert "document2_content.txt" in file_list
                assert "document2_metadata.json" in file_list
                assert len(file_list) == 4

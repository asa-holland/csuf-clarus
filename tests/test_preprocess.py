import pytest
import os
import tempfile
from unittest.mock import patch, mock_open, MagicMock

from clarus.steps.preprocess import (
    allowed_file,
    extract_pdf_text,
    extract_docx_text,
    extract_html_text,
    extract_txt_text,
    extract_text_from_file,
)


class TestAllowedFile:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("document.pdf", True),
            ("REPORT.PDF", True),
            ("file.pdf", True),
            ("document.docx", True),
            ("REPORT.DOCX", True),
            ("page.html", True),
            ("PAGE.HTML", True),
            ("site.htm", True),
            ("SITE.HTM", True),
            ("document.txt", True),
            ("README.TXT", True),
            ("image.jpg", False),
            ("script.js", False),
            ("data.csv", False),
            ("archive.zip", False),
            ("README", False),
            ("Makefile", False),
            ("", False),
            (None, False),
        ],
    )
    def test_should_validate_file_extensions(self, filename, expected):
        assert allowed_file(filename) is expected


class TestExtractPdfText:
    @patch("clarus.steps.preprocess.pdfminer.high_level.extract_text")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf content")
    def test_should_extract_pdf_text_successfully(self, mock_file, mock_extract):
        mock_extract.return_value = "Sample PDF content"

        text, metadata = extract_pdf_text("test.pdf")

        assert text == "Sample PDF content"
        assert metadata["format"] == "PDF"
        assert metadata["extraction_method"] == "pdfminer.six"
        assert metadata["encoding"] == "UTF-8"
        mock_file.assert_called_once_with("test.pdf", "rb")

    @patch("clarus.steps.preprocess.pdfminer.high_level.extract_text")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf content")
    def test_should_handle_pdf_extraction_error(self, mock_file, mock_extract):
        """Should handle PDF extraction errors gracefully."""
        mock_extract.side_effect = Exception("PDF parsing error")

        with pytest.raises(Exception, match="PDF extraction failed"):
            extract_pdf_text("corrupt.pdf")


class TestExtractDocxText:

    @patch("clarus.steps.preprocess.Document")
    def test_should_extract_docx_text_successfully(self, mock_document_class):
        mock_doc = MagicMock()
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Second paragraph"
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]

        mock_doc.core_properties.title = "Test Document"
        mock_doc.core_properties.author = "Test Author"
        mock_doc.core_properties.created = "2024-01-01"
        mock_doc.core_properties.modified = "2024-01-02"

        mock_document_class.return_value = mock_doc

        text, metadata = extract_docx_text("test.docx")

        assert text == "First paragraph\nSecond paragraph"
        assert metadata["format"] == "DOCX"
        assert metadata["extraction_method"] == "python-docx"
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        mock_document_class.assert_called_once_with("test.docx")

    @patch("clarus.steps.preprocess.Document")
    def test_should_handle_docx_extraction_error(self, mock_document_class):
        """Should handle DOCX extraction errors gracefully."""
        mock_document_class.side_effect = Exception("DOCX parsing error")

        with pytest.raises(Exception, match="DOCX extraction failed"):
            extract_docx_text("corrupt.docx")


class TestExtractHtmlText:

    @patch("clarus.steps.preprocess.readability.Document")
    @patch("clarus.steps.preprocess.BeautifulSoup")
    @patch("clarus.steps.preprocess.chardet.detect")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=b"<html><body>Test content</body></html>",
    )
    def test_should_extract_html_text_successfully(
        self, mock_file, mock_chardet, mock_bs, mock_readability
    ):
        mock_chardet.return_value = {"encoding": "utf-8", "confidence": 0.99}

        mock_readable_doc = MagicMock()
        mock_readable_doc.summary.return_value = (
            "<html><body>Clean content</body></html>"
        )
        mock_readable_doc.score.return_value = 50.0
        mock_readability.return_value = mock_readable_doc

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Clean content"
        mock_title = MagicMock()
        mock_title.get_text.return_value = "Test Page"
        mock_soup.find.return_value = mock_title
        mock_bs.return_value = mock_soup

        text, metadata = extract_html_text("test.html")

        assert text == "Clean content"
        assert metadata["format"] == "HTML"
        assert metadata["extraction_method"] == "readability-lxml + BeautifulSoup"
        assert metadata["encoding"] == "utf-8"
        assert metadata["title"] == "Test Page"
        assert metadata["readability_score"] == 50.0

    @patch("clarus.steps.preprocess.readability.Document")
    @patch("clarus.steps.preprocess.BeautifulSoup")
    @patch("clarus.steps.preprocess.chardet.detect")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=b"<html><body>Test content</body></html>",
    )
    def test_should_handle_html_extraction_error(
        self, mock_file, mock_chardet, mock_bs, mock_readability
    ):
        mock_chardet.return_value = {"encoding": "utf-8", "confidence": 0.99}
        mock_readability.side_effect = Exception("HTML parsing error")

        with pytest.raises(Exception, match="HTML extraction failed"):
            extract_html_text("corrupt.html")


class TestExtractTxtText:

    @patch("clarus.steps.preprocess.chardet.detect")
    @patch("builtins.open", new_callable=mock_open, read_data=b"Sample text content")
    def test_should_extract_txt_text_successfully(self, mock_file, mock_chardet):
        mock_chardet.return_value = {"encoding": "utf-8", "confidence": 0.95}

        text, metadata = extract_txt_text("test.txt")

        assert text == "Sample text content"
        assert metadata["format"] == "TXT"
        assert metadata["extraction_method"] == "direct read + chardet"
        assert metadata["encoding"] == "utf-8"
        assert metadata["encoding_confidence"] == 0.95
        assert metadata["line_count"] == 1
        assert metadata["word_count"] == 3
        assert metadata["character_count"] == 19

    @patch("clarus.steps.preprocess.chardet.detect")
    @patch("builtins.open", new_callable=mock_open, read_data=b"Sample text content")
    def test_should_handle_unicode_decode_error(self, mock_file, mock_chardet):
        mock_chardet.return_value = {"encoding": "utf-8", "confidence": 0.95}

        with patch(
            "builtins.open", mock_open(read_data=b"Invalid \xff\xfe unicode")
        ) as mock_file:
            text, metadata = extract_txt_text("test.txt")

            assert text is not None
            assert metadata["encoding"] == "utf-8 (fallback)"

    @patch("clarus.steps.preprocess.chardet.detect")
    @patch("builtins.open", new_callable=mock_open, read_data=b"Sample text content")
    def test_should_handle_txt_extraction_error(self, mock_file, mock_chardet):
        mock_file.side_effect = IOError("File not found")

        with pytest.raises(Exception, match="TXT extraction failed"):
            extract_txt_text("missing.txt")


class TestExtractTextFromFile:

    @pytest.mark.parametrize(
        "file_extension,extractor_function",
        [
            ("pdf", "extract_pdf_text"),
            ("docx", "extract_docx_text"),
            ("html", "extract_html_text"),
            ("htm", "extract_html_text"),
            ("txt", "extract_txt_text"),
        ],
    )
    @patch("clarus.steps.preprocess.extract_txt_text")
    @patch("clarus.steps.preprocess.extract_html_text")
    @patch("clarus.steps.preprocess.extract_docx_text")
    @patch("clarus.steps.preprocess.extract_pdf_text")
    def test_should_call_correct_extractor(
        self,
        mock_pdf,
        mock_docx,
        mock_html,
        mock_txt,
        file_extension,
        extractor_function,
    ):
        mock_pdf.return_value = ("PDF content", {"format": "PDF"})
        mock_docx.return_value = ("DOCX content", {"format": "DOCX"})
        mock_html.return_value = ("HTML content", {"format": "HTML"})
        mock_txt.return_value = ("TXT content", {"format": "TXT"})

        text, metadata = extract_text_from_file(
            f"test.{file_extension}", file_extension
        )

        if file_extension == "pdf":
            mock_pdf.assert_called_once_with("test.pdf")
            assert text == "PDF content"
        elif file_extension == "docx":
            mock_docx.assert_called_once_with("test.docx")
            assert text == "DOCX content"
        elif file_extension in ["html", "htm"]:
            mock_html.assert_called_once_with(f"test.{file_extension}")
            assert text == "HTML content"
        elif file_extension == "txt":
            mock_txt.assert_called_once_with("test.txt")
            assert text == "TXT content"

    @pytest.mark.parametrize(
        "unsupported_extension", ["jpg", "png", "mp3", "zip", "csv"]
    )
    def test_should_raise_error_for_unsupported_formats(self, unsupported_extension):
        with pytest.raises(Exception, match="Unsupported file format"):
            extract_text_from_file(
                f"test.{unsupported_extension}", unsupported_extension
            )

    @pytest.mark.parametrize(
        "case_variant", ["PDF", "Pdf", "pdf", "DOCX", "Docx", "docx"]
    )
    @patch("clarus.steps.preprocess.extract_pdf_text")
    @patch("clarus.steps.preprocess.extract_docx_text")
    def test_should_handle_case_insensitive_extensions(
        self, mock_docx, mock_pdf, case_variant
    ):
        if case_variant.lower() == "pdf":
            mock_pdf.return_value = ("content", {"format": "PDF"})
            extract_text_from_file(f"test.{case_variant}", case_variant)
            mock_pdf.assert_called_once_with(f"test.{case_variant}")
        else:
            mock_docx.return_value = ("content", {"format": "DOCX"})
            extract_text_from_file(f"test.{case_variant}", case_variant)
            mock_docx.assert_called_once_with(f"test.{case_variant}")


class TestIntegration:

    @pytest.mark.parametrize(
        "content,expected_lines,expected_words,expected_chars",
        [
            ("This is a test file.\nWith multiple lines.", 2, 8, 41),
            ("Single line", 1, 2, 11),
            ("", 1, 0, 0),  # Empty file still has 1 line
            ("Word1 Word2 Word3", 1, 3, 17),
        ],
    )
    def test_should_process_txt_files_with_various_content(
        self, content, expected_lines, expected_words, expected_chars
    ):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            text, metadata = extract_txt_text(temp_path)

            assert text == content
            assert metadata["format"] == "TXT"
            assert metadata["line_count"] == expected_lines
            assert metadata["word_count"] == expected_words
            assert metadata["character_count"] == expected_chars
        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize(
        "encoding,content",
        [
            ("utf-8", "UTF-8 content: café résumé naïve"),
            ("ascii", "ASCII content only"),
            ("utf-16", "UTF-16 content"),
        ],
    )
    def test_should_handle_different_encodings(self, encoding, content):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding=encoding
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            text, metadata = extract_txt_text(temp_path)

            assert content in text
            assert metadata["format"] == "TXT"
        finally:
            os.unlink(temp_path)

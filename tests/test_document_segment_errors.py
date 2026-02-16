import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from clarus.analysis.document_processor import (
    DocumentProcessor,
    DocumentSegment,
    ElementType,
)


class TestDocumentSegmentErrors:

    def test_document_segment_should_have_errors_attribute(self):
        segment = DocumentSegment(
            text="Test segment", element_type=ElementType.NORMATIVE, confidence=0.8
        )

        assert hasattr(
            segment, "errors"
        ), "DocumentSegment should have errors attribute"
        assert segment.errors == [], "Default errors should be empty list"

    def test_document_segment_errors_should_be_list(self):
        segment = DocumentSegment(
            text="Test segment", element_type=ElementType.NORMATIVE, confidence=0.8
        )

        segment.errors.append("Test error message")
        assert len(segment.errors) == 1
        assert segment.errors[0] == "Test error message"

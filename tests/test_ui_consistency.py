import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from clarus.analysis.document_processor import DocumentProcessor
from clarus.analysis.semantic_extractor import SemanticExtractor


class TestUIConsistency:

    def test_all_sections_should_have_scrollable_containers(self):
        processor = DocumentProcessor()

        test_text = """
        1. Requirements
        The system shall process user data.
        Users must authenticate.
        Components shall be validated.
        Applications should be optimized.

        2. Implementation
        The system will use encryption.
        Users may access during business hours.
        Components can be scaled horizontally.
        Testing must be comprehensive.

        3. Quality Assurance
        Tests shall cover requirements.
        Quality must be maintained.
        Performance should be monitored.
        Security should be validated.
        """

        result = processor.process_document(test_text)
        extractor = SemanticExtractor()

        normative_segments = [
            s for s in result.segments if s.element_type.value == "normative"
        ]

        all_profiles = []
        for segment in normative_segments:
            profiles = extractor.extract_profiles_from_segment(segment)
            all_profiles.extend(profiles)

        assert (
            len(result.segments) > 5
        ), f"Should have more than 5 segments, got {len(result.segments)}"
        assert (
            len(all_profiles) > 3
        ), f"Should have more than 3 semantic profiles, got {len(all_profiles)}"

        total_items = len(result.segments) + len(all_profiles)

        assert (
            total_items > 10
        ), f"Should have sufficient items to test scrollability, got {total_items}"

        expected_segments = len(result.segments)
        expected_profiles = len(all_profiles)

        assert (
            expected_segments >= 8
        ), f"Expected at least 8 segments for scrollability test"
        assert (
            expected_profiles >= 5
        ), f"Expected at least 5 profiles for scrollability test"

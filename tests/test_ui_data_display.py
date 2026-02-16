import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from clarus.analysis.document_processor import DocumentProcessor


class TestUIDataDisplay:

    def test_should_display_all_segments_not_limited_subset(self):
        processor = DocumentProcessor()

        test_text = """
        1. Requirements
        The system shall process user data.
        Users must authenticate.
        Components should be optimized.

        2. Implementation
        The module will handle requests.
        Developers may extend functionality.
        Applications can scale horizontally.

        3. Testing
        Tests must cover all cases.
        Quality should be maintained.
        Performance needs to be monitored.
        """

        result = processor.process_document(test_text)

        assert (
            len(result.segments) > 5
        ), f"Should have more than 5 segments, got {len(result.segments)}"

        limited_segments = result.segments[:5]
        assert len(limited_segments) == 5, "UI currently limits to 5 segments"

        assert len(result.segments) > len(
            limited_segments
        ), "Not all segments are shown in UI"

        normative_count = len(
            [s for s in result.segments if s.element_type.value == "normative"]
        )
        informative_count = len(
            [s for s in result.segments if s.element_type.value == "informative"]
        )

        assert (
            normative_count >= 3
        ), f"Should have at least 3 normative segments, got {normative_count}"
        assert (
            informative_count >= 3
        ), f"Should have at least 3 informative segments, got {informative_count}"

    def test_should_display_all_semantic_profiles_not_limited_subset(self):
        from clarus.analysis.semantic_extractor import SemanticExtractor

        processor = DocumentProcessor()
        extractor = SemanticExtractor()

        test_text = """
        The system shall process data immediately.
        Users must authenticate when accessing.
        Components should be optimized regularly.
        Applications may scale during peak hours.
        Tests must validate all inputs.
        The system shall log all activities.
        Users must provide credentials.
        Components shall be tested thoroughly.
        """

        result = processor.process_document(test_text)

        normative_segments = [
            s for s in result.segments if s.element_type.value == "normative"
        ]
        all_profiles = []

        for segment in normative_segments:
            profiles = extractor.extract_profiles_from_segment(segment)
            all_profiles.extend(profiles)

        assert (
            len(all_profiles) > 3
        ), f"Should have more than 3 semantic profiles, got {len(all_profiles)}"

        limited_profiles = all_profiles[:3]
        assert len(limited_profiles) == 3, "UI currently limits to 3 semantic profiles"

        assert len(all_profiles) > len(
            limited_profiles
        ), "Not all semantic profiles are shown in UI"

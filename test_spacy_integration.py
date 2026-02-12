import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clarus.analysis.document_processor import DocumentProcessor, ElementType


def test_spacy_integration():
    processor = DocumentProcessor()
    test_document = """1. System Requirements

1.1 Authentication
The system shall authenticate users before access.
If the temperature exceeds 90°C, the operator must immediately initiate the emergency cooling sequence.

1.2 Data Processing
Users should verify credentials within 5 minutes.
The system may log activities for security purposes.

2. Examples
For example, consider the following case:
Note: This is an important consideration."""
    result = processor.process_document(test_document)

    print("=== Document Processing Results ===")
    print(f"Total segments: {result.processing_stats['total_segments']}")
    print(f"Normative segments: {result.processing_stats['normative_segments']}")
    print(f"Informative segments: {result.processing_stats['informative_segments']}")
    print()

    print("=== Semantic Anchors Analysis ===")
    for segment in result.segments:
        if segment.element_type == ElementType.NORMATIVE and segment.semantic_anchors:
            print(f"\nSegment: {segment.text}")
            anchors = segment.semantic_anchors
            print(f"  Subject: {anchors.subject}")
            print(f"  Modality: {anchors.modality}")
            print(f"  Object: {anchors.object}")
            print(f"  Temporal: {anchors.temporal}")
            print(f"  Condition: {anchors.condition}")
            print(f"  Negation: {anchors.negation}")
            print(f"  Confidence: {anchors.confidence:.2f}")

    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    test_spacy_integration()

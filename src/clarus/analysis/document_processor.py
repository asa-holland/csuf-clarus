import re
import spacy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ElementType(Enum):
    """Document element types based on ISO/IEC directives"""

    NORMATIVE = "normative"  # Requirements/rules (shall/must provisions)
    INFORMATIVE = "informative"  # Context, examples, notes
    UNKNOWN = "unknown"  # Cannot be determined


@dataclass
class DocumentSegment:
    """Represents a segmented portion of the document"""

    text: str
    element_type: ElementType
    section_number: Optional[str] = None
    heading: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    confidence: float = 0.0
    semantic_anchors: Optional["SemanticAnchor"] = None
    errors: List[Dict] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Result of document processing with segmented elements"""

    original_text: str
    segments: List[DocumentSegment]
    metadata: Dict[str, any]
    processing_stats: Dict[str, int]


@dataclass
class SemanticAnchor:
    condition: Optional[str] = None
    subject: Optional[str] = None
    modality: Optional[str] = None
    object: Optional[str] = None
    temporal: Optional[str] = None
    negation: Optional[str] = None
    confidence: float = 0.0


class ErrorType(Enum):
    UNDEFINED_TERM = "Undefined Term"
    SYNONYM_INCONSISTENCY = "Synonym Inconsistency"
    DIRECT_CONTRADICTION = "Direct Contradiction"
    MODAL_INCONSISTENCY = "Modal Inconsistency"
    HIDDEN_NORMATIVE = "Hidden Normative"
    VAGUE_QUALIFIERS = "Vague Qualifiers"
    AMBIGUOUS_REFERENT = "Ambiguous Referent"
    MISSING_TEMPORAL = "Missing Temporal Anchor"


class DocumentProcessor:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Warning: SpaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        self.modality_verbs = {
            "shall",
            "must",
            "should",
            "may",
            "can",
            "will",
            "could",
            "would",
        }
        self.negation_words = {"not", "no", "never", "none", "without"}
        self.temporal_indicators = {
            "immediately",
            "promptly",
            "within",
            "before",
            "after",
            "when",
            "if",
        }
        self.condition_indicators = {
            "if",
            "when",
            "unless",
            "provided that",
            "in case of",
        }

        self.normative_patterns = [
            r"\bshall\b",
            r"\bmust\b",
            r"\bshall not\b",
            r"\bmust not\b",
            r"\bis required to\b",
            r"\bis prohibited from\b",
        ]

        self.informative_patterns = [
            r"\bshould\b",
            r"\bmay\b",
            r"\bcan\b",
            r"\bfor example\b",
            r"\be\.g\.\b",
            r"\bi\.e\.\b",
            r"\bnote\b",
            r"\bexample\b",
        ]

        self.heading_patterns = [
            r"^(\d+\.?\d*)\s+(.+)$",
            r"^([A-Z]+[A-Z\s]*)$",
            r"^([IVX]+\.?\s*.+)$",
        ]
        self.normative_regex = re.compile(
            "|".join(self.normative_patterns), re.IGNORECASE
        )
        self.informative_regex = re.compile(
            "|".join(self.informative_patterns), re.IGNORECASE
        )
        self.heading_regex = [
            re.compile(pattern, re.MULTILINE) for pattern in self.heading_patterns
        ]

    def extract_semantic_anchors(self, text: str) -> SemanticAnchor:
        if not self.nlp:
            return SemanticAnchor()

        doc = self.nlp(text)
        anchors = SemanticAnchor()

        for token in doc:
            if token.text.lower() in self.modality_verbs:
                anchors.modality = token.text
                break

        for token in doc:
            if "subj" in token.dep_ and not anchors.subject:
                anchors.subject = token.text

            elif "obj" in token.dep_ and not anchors.object:
                anchors.object = token.text

            elif token.text.lower() in self.negation_words:
                anchors.negation = token.text

        for token in doc:
            if token.text.lower() in self.temporal_indicators:
                anchors.temporal = token.text
                break

        if not anchors.temporal:
            for ent in doc.ents:
                if ent.label_ in ["TIME", "DATE", "DURATION"]:
                    anchors.temporal = ent.text
                    break

        for condition_indicator in self.condition_indicators:
            if condition_indicator in text.lower():
                condition_start = text.lower().find(condition_indicator)
                if condition_start != -1:
                    condition_end = text.find(",", condition_start)
                    if condition_end == -1:
                        condition_end = text.find(".", condition_start)
                    if condition_end == -1:
                        condition_end = len(text)
                    anchors.condition = text[condition_start:condition_end].strip()
                    break

        found_components = sum(
            [
                1
                for component in [
                    anchors.subject,
                    anchors.object,
                    anchors.modality,
                    anchors.temporal,
                    anchors.condition,
                    anchors.negation,
                ]
                if component is not None
            ]
        )
        anchors.confidence = found_components / 6.0

        return anchors

    def process_document(
        self, text: str, metadata: Optional[Dict] = None
    ) -> ProcessedDocument:
        if metadata is None:
            metadata = {}

        lines = text.split("\n")
        segments = []
        current_segment = None
        line_number = 0

        for line in lines:
            line_number += 1
            stripped_line = line.strip()

            if not stripped_line:
                if current_segment:
                    current_segment.end_line = line_number - 1
                    segments.append(current_segment)
                    current_segment = None
                continue

            heading_info = self._identify_heading(stripped_line)
            element_type = self._classify_line(stripped_line)

            should_start_new = False

            if heading_info:
                should_start_new = True
            elif current_segment is None:
                should_start_new = True
            elif (
                current_segment.element_type != element_type
                and not self._is_continuation(stripped_line)
            ):
                should_start_new = True
            elif self._should_start_new_segment(current_segment, stripped_line):
                should_start_new = True

            if should_start_new:
                if current_segment:
                    current_segment.end_line = line_number - 1
                    segments.append(current_segment)

                current_segment = DocumentSegment(
                    text=stripped_line,
                    element_type=element_type,
                    section_number=heading_info[0] if heading_info else None,
                    heading=heading_info[1] if heading_info else None,
                    start_line=line_number,
                    end_line=line_number,
                    confidence=self._calculate_confidence(stripped_line, element_type),
                )
            else:
                current_segment.text += "\n" + stripped_line
                current_segment.end_line = line_number
                current_segment.confidence = self._calculate_confidence(
                    current_segment.text, current_segment.element_type
                )

        if current_segment:
            current_segment.end_line = line_number
            segments.append(current_segment)

        for segment in segments:
            if segment.element_type == ElementType.NORMATIVE:
                segment.semantic_anchors = self.extract_semantic_anchors(segment.text)

        stats = self._calculate_stats(segments)

        return ProcessedDocument(
            original_text=text,
            segments=segments,
            metadata=metadata,
            processing_stats=stats,
        )

    def _should_start_new_segment(
        self, current_segment: DocumentSegment, new_line: str
    ) -> bool:
        if self._is_sentence_complete(current_segment.text):
            if not self._is_continuation(new_line):
                return True

        return False

    def _is_continuation(self, line: str) -> bool:
        continuation_words = [
            "and",
            "or",
            "but",
            "to",
            "for",
            "with",
            "without",
            "as",
            "by",
            "at",
            "in",
            "on",
            "from",
            "of",
            "while",
            "since",
            "because",
            "although",
            "though",
        ]

        words = line.lower().split()
        if words and words[0] in continuation_words:
            return True

        if line and line[0].islower():
            return True

        return False

    def _is_sentence_complete(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped or stripped[-1] not in ".!?":
            return False

        lines = text.split("\n")
        if len(lines) > 1:
            last_line = lines[-1].strip()
            if self._is_continuation(last_line):
                return False

        return True

    def _identify_heading(self, line: str) -> Optional[Tuple[Optional[str], str]]:
        for pattern in self.heading_regex:
            match = pattern.match(line)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1), match.group(2)
                else:
                    return None, match.group(1)
        return None

    def _classify_line(self, line: str) -> ElementType:
        if self.normative_regex.search(line):
            return ElementType.NORMATIVE

        if self.informative_regex.search(line):
            return ElementType.INFORMATIVE

        informative_keywords = ["example", "note", "remark", "comment"]
        if any(keyword in line.lower() for keyword in informative_keywords):
            return ElementType.INFORMATIVE

        return ElementType.UNKNOWN

    def _calculate_confidence(self, text: str, element_type: ElementType) -> float:
        if element_type == ElementType.NORMATIVE:
            normative_matches = len(self.normative_regex.findall(text))
            informative_matches = len(self.informative_regex.findall(text))

            if normative_matches > 0 and informative_matches == 0:
                return 0.9
            elif normative_matches > informative_matches:
                return 0.7
            else:
                return 0.3

        elif element_type == ElementType.INFORMATIVE:
            informative_matches = len(self.informative_regex.findall(text))
            normative_matches = len(self.normative_regex.findall(text))

            if informative_matches > 0 and normative_matches == 0:
                return 0.8
            elif informative_matches > normative_matches:
                return 0.6
            else:
                return 0.3

        return 0.1

    def _calculate_stats(self, segments: List[DocumentSegment]) -> Dict[str, int]:
        stats = {
            "total_segments": len(segments),
            "normative_segments": sum(
                1 for s in segments if s.element_type == ElementType.NORMATIVE
            ),
            "informative_segments": sum(
                1 for s in segments if s.element_type == ElementType.INFORMATIVE
            ),
            "unknown_segments": sum(
                1 for s in segments if s.element_type == ElementType.UNKNOWN
            ),
            "high_confidence_segments": sum(1 for s in segments if s.confidence >= 0.7),
            "medium_confidence_segments": sum(
                1 for s in segments if 0.3 <= s.confidence < 0.7
            ),
            "low_confidence_segments": sum(1 for s in segments if s.confidence < 0.3),
        }
        return stats

    def get_normative_segments(
        self, processed_doc: ProcessedDocument
    ) -> List[DocumentSegment]:
        return [
            s for s in processed_doc.segments if s.element_type == ElementType.NORMATIVE
        ]

    def get_informative_segments(
        self, processed_doc: ProcessedDocument
    ) -> List[DocumentSegment]:
        return [
            s
            for s in processed_doc.segments
            if s.element_type == ElementType.INFORMATIVE
        ]

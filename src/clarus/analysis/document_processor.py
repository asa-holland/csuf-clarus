import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple


class ElementType(Enum):
    """Document element types based on ISO/IEC directives"""

    NORMATIVE = "normative"  # Requirements/rules (shall/must provisions)
    INFORMATIVE = "informative"  # Context, examples, notes
    UNKNOWN = "unknown"  # Cannot be determined


@dataclass
class DocumentSegment:

    text: str
    element_type: ElementType
    section_number: Optional[str] = None
    heading: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    confidence: float = 0.0
    semantic_anchors: Optional["SemanticAnchor"] = None


@dataclass
class SemanticAnchor:

    subject: Optional[str] = None
    object: Optional[str] = None
    modality: Optional[str] = None
    temporal: Optional[str] = None
    condition: Optional[str] = None
    negation: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ProcessedDocument:

    original_text: str
    segments: List[DocumentSegment]
    metadata: Dict = field(default_factory=dict)
    processing_stats: Dict = field(default_factory=dict)


class DocumentProcessor:

    def __init__(self):
        import spacy
        from spacy.cli import download

        try:
            self.nlp = spacy.load("en_core_web_sm")
            print(f"✅ SpaCy model 'en_core_web_sm' loaded successfully")

        except OSError as e:
            print(f"❌ SpaCy model not found: {e}")
            print("🔄 Attempting to download SpaCy model...")

            try:
                download("en_core_web_sm")
                print("✅ SpaCy model downloaded successfully")

                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print(f"✅ SpaCy model loaded successfully after download")
                except OSError as e2:
                    print(f"❌ Still failed to load SpaCy model after download: {e2}")
                    self.nlp = None

            except Exception as download_error:
                print(f"❌ Failed to download SpaCy model: {download_error}")
                self.nlp = None

        except Exception as e:
            print(f"❌ Unexpected error loading SpaCy: {e}")
            self.nlp = None

        if self.nlp is not None:
            self._validate_spacy_functionality()

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

    def _validate_spacy_functionality(self):
        try:
            test_doc = self.nlp("The system shall process data.")

            if len(test_doc) == 0:
                print("❌ SpaCy validation failed: No tokens produced")
                self.nlp = None
                return

            has_pos = hasattr(test_doc[0], "pos_") if test_doc else False
            has_dep = hasattr(test_doc[0], "dep_") if test_doc else False

            if not has_pos or not has_dep:
                print("❌ SpaCy validation failed: Missing linguistic features")
                self.nlp = None
                return

            print("✅ SpaCy validation passed: All features available")

        except Exception as e:
            print(f"❌ SpaCy validation failed: {e}")
            self.nlp = None

    def _classify_line(self, line: str) -> ElementType:
        if not self.nlp:
            return self._classify_line_regex(line)

        doc = self.nlp(line)

        informative_lemmas = [
            "example",
            "note",
            "remark",
            "comment",
            "illustrate",
            "demonstrate",
            "section",
            "detail",
            "see",
            "reference",
        ]
        informative_found = any(
            token.lemma_.lower() in informative_lemmas for token in doc
        )

        if informative_found:
            return ElementType.INFORMATIVE

        if any(token.text == "?" for token in doc):
            return ElementType.INFORMATIVE

        modality_tokens = []
        for token in doc:
            if token.lemma_.lower() in self.modality_verbs:
                negated = False
                for child in token.children:
                    if child.lemma_.lower() in self.negation_words:
                        negated = True
                        break
                modality_tokens.append(
                    (token.lemma_.lower(), "negated" if negated else "positive")
                )

        if modality_tokens:
            positive_modalities = [t[0] for t in modality_tokens if t[1] == "positive"]
            negated_modalities = [t[0] for t in modality_tokens if t[1] == "negated"]

            strong_normative = any(
                lemma in ["shall", "must"] for lemma in positive_modalities
            )

            if strong_normative and not negated_modalities:
                return ElementType.NORMATIVE

            elif negated_modalities:
                return ElementType.INFORMATIVE

            elif any(
                lemma in ["should", "may", "can", "could"]
                for lemma in positive_modalities
            ):
                return ElementType.INFORMATIVE

            return ElementType.NORMATIVE

        negation_found = any(
            token.lemma_.lower() in self.negation_words for token in doc
        )
        imperative_found = False
        for sent in doc.sents:
            if sent.root and sent.root.tag_ in [
                "VB",
                "VBP",
            ]:
                imperative_found = True
                break

        if imperative_found and not negation_found and self._is_clear_imperative(doc):
            return ElementType.NORMATIVE

        return self._classify_line_regex(line)

    def _is_clear_imperative(self, doc) -> bool:
        imperative_indicators = [
            "shall",
            "must",
            "should",
            "ensure",
            "verify",
            "validate",
        ]

        for token in doc:
            if token.lemma_.lower() in imperative_indicators:
                return True

        return False

    def _classify_line_regex(self, line: str) -> ElementType:
        if self.normative_regex.search(line):
            return ElementType.NORMATIVE

        if self.informative_regex.search(line):
            return ElementType.INFORMATIVE

        informative_keywords = ["example", "note", "remark", "comment"]
        if any(keyword in line.lower() for keyword in informative_keywords):
            return ElementType.INFORMATIVE

        return ElementType.UNKNOWN

    def _calculate_confidence(self, text: str, element_type: ElementType) -> float:
        if not self.nlp:
            return self._calculate_confidence_regex(text, element_type)

        doc = self.nlp(text)

        if element_type == ElementType.NORMATIVE:
            strong_modalities = ["shall", "must", "will"]
            weak_modalities = ["should", "may", "can", "could", "would"]

            strong_found = any(
                token.lemma_.lower() in strong_modalities for token in doc
            )
            weak_found = any(token.lemma_.lower() in weak_modalities for token in doc)
            negation_found = any(
                token.lemma_.lower() in self.negation_words for token in doc
            )

            if strong_found and not weak_found and not negation_found:
                return 0.95  # High confidence: strong normative
            elif strong_found and not negation_found:
                return 0.85  # Good confidence: normative with some weak modalities
            elif weak_found and not negation_found:
                return 0.65  # Medium confidence: weak modalities
            elif negation_found:
                return 0.45  # Low confidence: negated normative
            else:
                return 0.75  # Decent confidence: mixed signals

        elif element_type == ElementType.INFORMATIVE:
            informative_lemmas = [
                "example",
                "note",
                "remark",
                "comment",
                "illustrate",
                "section",
                "detail",
                "see",
            ]
            question_found = any(token.text == "?" for token in doc)

            clear_informative = any(
                token.lemma_.lower() in informative_lemmas for token in doc
            )
            if clear_informative or question_found:
                return 0.90  # High confidence: clear informative
            else:
                return 0.60  # Medium confidence: inferred informative

        return 0.40  # Low confidence: unknown

    def _calculate_confidence_regex(
        self, text: str, element_type: ElementType
    ) -> float:
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
            elif current_segment.element_type != element_type or (
                current_segment.element_type == element_type
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

    def _identify_heading(self, line: str) -> Optional[Tuple[str, str]]:
        for i, pattern in enumerate(self.heading_regex):
            match = pattern.search(line)
            if match:
                if "section" in match.groupdict():
                    return match.group("section"), match.group("heading")
                elif i == 0:  # Numbered heading pattern: ^(\d+\.?\d*)\s+(.+)$
                    return match.group(1), match.group(2)
                elif i == 1:  # All-caps pattern: ^([A-Z]+[A-Z\s]*)$
                    return None, match.group(1)
                elif i == 2:  # Roman numeral pattern: ^([IVX]+\.?\s*.+)$
                    return match.group(1), (
                        match.group(2) if len(match.groups()) > 1 else match.group(1)
                    )
        return None

    def _is_continuation(self, line: str) -> bool:
        return not (line[0].isupper() or line[0].isdigit() or line.startswith("("))

    def _should_start_new_segment(
        self, current_segment: DocumentSegment, line: str
    ) -> bool:
        return len(current_segment.text.split()) > 20 and len(line.split()) > 5

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
                1 for s in segments if 0.4 <= s.confidence < 0.7
            ),
            "low_confidence_segments": sum(1 for s in segments if s.confidence < 0.4),
        }
        return stats

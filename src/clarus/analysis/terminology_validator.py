import re
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(
        "Warning: transformers not available. Install with: pip install transformers sentence-transformers torch"
    )

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class TermClassification:

    term: str
    is_domain_term: bool
    is_common_english: bool
    confidence: float
    semantic_embedding: Optional[np.ndarray] = None
    classification_reason: str = ""


@dataclass
class DefinitionDetection:

    term: str
    has_definition: bool
    definition_text: Optional[str] = None
    confidence: float = 0.0
    context_sentence: Optional[str] = None


@dataclass
class ValidationResult:

    term: str
    normalized_term: str
    is_defined: bool
    classification: TermClassification
    definition_detection: Optional[DefinitionDetection] = None
    semantic_similarity_to_defined: float = 0.0
    suggested_definition: Optional[str] = None
    context_excerpt: Optional[str] = None


@dataclass
class ValidationReport:

    defined_terms: List[ValidationResult]
    undefined_terms: List[ValidationResult]
    common_english_terms: List[ValidationResult]
    potential_domain_terms: List[ValidationResult]
    statistics: Dict[str, Union[int, float]]
    semantic_clusters: Optional[Dict[str, List[str]]] = None


class TerminologyValidator:
    def __init__(
        self,
        glossary: Optional[Dict[str, str]] = None,
        term_extraction_model: str = "roberta-base",
        definition_model: str = "roberta-base",
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.glossary = glossary or {}
        self._term_index = self._build_term_index()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.term_extraction_tokenizer = None
        self.term_extraction_model = None
        self.definition_tokenizer = None
        self.definition_model = None
        self.semantic_model = None
        self.nlp = None

        if TRANSFORMERS_AVAILABLE:
            self._load_models(term_extraction_model, definition_model, semantic_model)

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print(
                    "Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm"
                )

    def _load_models(self, term_model: str, def_model: str, semantic_model: str):
        try:
            print("Loading transformer models...")

            self.term_extraction_tokenizer = AutoTokenizer.from_pretrained(term_model)
            self.term_extraction_model = (
                AutoModelForTokenClassification.from_pretrained(term_model)
            )
            self.term_extraction_model.to(self.device)
            self.term_extraction_model.eval()

            self.definition_tokenizer = AutoTokenizer.from_pretrained(def_model)
            self.definition_model = AutoModelForSequenceClassification.from_pretrained(
                def_model
            )
            self.definition_model.to(self.device)
            self.definition_model.eval()

            self.semantic_model = SentenceTransformer(semantic_model)

            print("Models loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load transformer models: {e}")
            self.term_extraction_tokenizer = None
            self.term_extraction_model = None
            self.definition_tokenizer = None
            self.definition_model = None
            self.semantic_model = None

    def add_glossary_terms(self, terms: Dict[str, str]) -> None:
        self.glossary.update(terms)
        self._term_index = self._build_term_index()

    def _extract_definitions_from_text(self, text: str) -> Dict[str, str]:
        """Extract definitions from text using rule-based patterns"""
        definitions = {}

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.search(
                r"([A-Za-z][A-Za-z0-9\s\-\.\d]{2,25}?)\s+is\s+([^.!?]+?)(?=[.!?]|$)",
                line,
                re.IGNORECASE,
            )
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if (
                    term
                    and definition
                    and len(term) <= 25
                    and len(definition.strip()) > len(term.strip())
                ):
                    definitions[term.lower()] = definition
                continue

            match = re.search(
                r"([A-Za-z][A-Za-z0-9\s\-\.\d]{2,25}?)\s*\(\s*([A-Za-z][A-Za-z0-9\s\-\.\d]{5,80}?)\s*\)(?=[.!?]|$)",
                line,
                re.IGNORECASE,
            )
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if (
                    term
                    and definition
                    and len(term) <= 25
                    and len(definition.strip()) > len(term.strip())
                ):
                    definitions[term.lower()] = definition
                continue

            match = re.search(
                r"([A-Za-z][A-Za-z0-9\s\-\.\d]{2,25}?)\s*-\s*([^.!?]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if (
                    term
                    and definition
                    and len(term) <= 25
                    and len(definition.strip()) > len(term.strip())
                ):
                    definitions[term.lower()] = definition
                continue

            match = re.search(
                r"([A-Za-z][A-Za-z0-9\s\-\.\d]{2,25}?):\s*([^.!?]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if (
                    term
                    and definition
                    and len(term) <= 25
                    and len(definition.strip()) > len(term.strip())
                ):
                    definitions[term.lower()] = definition
                continue

        return definitions

    def _build_term_index(self) -> Dict[str, str]:
        index = {}
        for term, definition in self.glossary.items():
            term_lower = term.lower()
            index[term_lower] = term
            if term.endswith("s") and len(term) > 3:
                singular = term[:-1].lower()
                index[singular] = term
            else:
                plural = (term + "s").lower()
                index[plural] = term

        return index

    def _get_term_embedding(self, term: str) -> Optional[np.ndarray]:
        if self.semantic_model is None:
            return None
        try:
            return self.semantic_model.encode(term, convert_to_numpy=True)
        except Exception as e:
            print(f"Error getting embedding for '{term}': {e}")
            return None

    def _classify_term_with_transformer(
        self, term: str, context: Optional[str] = None
    ) -> TermClassification:
        if self.term_extraction_model is None:
            return self._classify_term_rule_based(term, context)

        try:
            if context:
                text = f"{context} [SEP] {term}"
            else:
                text = term

            inputs = self.term_extraction_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.term_extraction_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = torch.max(predictions).item()
                is_domain_term = confidence > 0.7

            embedding = self._get_term_embedding(term)

            is_common_english = self._is_common_english_semantic(term, embedding)

            classification_reason = "transformer_classification"
            if is_domain_term:
                classification_reason = "domain_term_detected"
            elif is_common_english:
                classification_reason = "common_english_detected"

            return TermClassification(
                term=term,
                is_domain_term=is_domain_term and not is_common_english,
                is_common_english=is_common_english,
                confidence=confidence,
                semantic_embedding=embedding,
                classification_reason=classification_reason,
            )

        except Exception as e:
            print(f"Error in transformer classification for '{term}': {e}")
            return self._classify_term_rule_based(term, context)

    def _classify_term_rule_based(
        self, term: str, context: Optional[str] = None
    ) -> TermClassification:
        term_lower = term.lower()

        common_english = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "where",
            "while",
            "because",
            "since",
            "until",
            "although",
            "though",
            "unless",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "from",
            "by",
            "about",
            "as",
            "into",
            "like",
            "through",
            "after",
            "over",
            "between",
            "out",
            "against",
            "during",
            "without",
            "before",
            "under",
            "around",
            "among",
            "throughout",
            "behind",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "shall",
            "use",
            "used",
            "using",
            "make",
            "made",
            "take",
            "took",
            "taken",
            "give",
            "gave",
            "given",
            "come",
            "came",
            "become",
            "time",
            "person",
            "year",
            "way",
            "day",
            "man",
            "thing",
            "woman",
            "life",
            "child",
            "world",
            "school",
            "state",
            "family",
            "student",
            "group",
            "country",
            "problem",
            "hand",
            "part",
            "place",
            "case",
            "week",
            "company",
            "system",
            "program",
            "question",
            "work",
            "government",
            "number",
            "night",
            "point",
            "home",
            "water",
            "room",
            "mother",
            "area",
            "money",
            "story",
            "fact",
            "month",
            "lot",
            "right",
            "study",
            "book",
            "eye",
            "job",
            "word",
            "business",
            "issue",
            "side",
            "kind",
            "head",
            "house",
            "service",
            "friend",
            "father",
            "power",
            "hour",
            "game",
            "line",
            "end",
            "member",
            "law",
            "car",
            "city",
            "data",
            "information",
            "process",
            "method",
            "approach",
            "technique",
            "function",
            "procedure",
            "operation",
            "activity",
            "task",
            "step",
            "phase",
            "stage",
            "level",
            "type",
            "kind",
            "form",
            "format",
            "structure",
            "model",
            "framework",
            "design",
            "implementation",
            "development",
            "analysis",
            "evaluation",
            "assessment",
            "testing",
            "validation",
            "verification",
        }

        is_common = term_lower in common_english

        is_domain = (
            (term.isupper() and len(term) >= 2)  # Acronym
            or (re.search(r"[a-z][A-Z]", term))  # CamelCase
            or (
                (" " in term and len(term.split()) >= 2 and not is_common)
            )  # Multi-word
            or (
                len(term) > 4
                and any(
                    suffix in term_lower
                    for suffix in [
                        "tion",
                        "ment",
                        "ness",
                        "ity",
                        "er",
                        "or",
                        "ism",
                        "ist",
                        "ive",
                        "able",
                        "ible",
                        "al",
                        "ial",
                        "ic",
                        "ical",
                        "ous",
                        "ious",
                        "ful",
                        "less",
                        "ize",
                        "ise",
                        "fy",
                        "ate",
                        "en",
                        "ify",
                    ]
                )
            )
        )

        confidence = 0.8 if is_domain else 0.6 if is_common else 0.4

        return TermClassification(
            term=term,
            is_domain_term=is_domain and not is_common,
            is_common_english=is_common,
            confidence=confidence,
            classification_reason="rule_based_fallback",
        )

    def _is_common_english_semantic(
        self, term: str, embedding: Optional[np.ndarray]
    ) -> bool:
        if embedding is None:
            return False

        common_references = [
            "the",
            "and",
            "have",
            "will",
            "time",
            "person",
            "way",
            "day",
            "man",
            "thing",
            "use",
            "make",
            "work",
            "part",
            "take",
            "get",
            "place",
            "live",
            "back",
            "only",
        ]

        try:
            common_embeddings = self.semantic_model.encode(
                common_references, convert_to_numpy=True
            )
            similarities = np.dot(embedding, common_embeddings.T)
            max_similarity = np.max(similarities)

            return max_similarity > 0.7

        except Exception as e:
            print(f"Error in semantic common English detection: {e}")
            return False

    def _detect_definition_with_transformer(
        self, term: str, context: str
    ) -> DefinitionDetection:
        if self.definition_model is None:
            return self._detect_definition_rule_based(term, context)

        try:
            text = f"Is '{term}' defined in: {context}"

            inputs = self.definition_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.definition_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                has_def = torch.argmax(predictions, dim=-1).item() == 1
                confidence = torch.max(predictions).item()

            definition_text = None
            if has_def:
                definition_text = self._extract_definition_from_context(term, context)

            return DefinitionDetection(
                term=term,
                has_definition=has_def,
                definition_text=definition_text,
                confidence=confidence,
                context_sentence=context,
            )

        except Exception as e:
            print(f"Error in transformer definition detection for '{term}': {e}")
            return self._detect_definition_rule_based(term, context)

    def _detect_definition_rule_based(
        self, term: str, context: str
    ) -> DefinitionDetection:
        definition_patterns = [
            rf"{term}\s+is\s+([^.!?]+)",
            rf"{term}\s+means\s+([^.!?]+)",
            rf"{term}\s+refers\s+to\s+([^.!?]+)",
            rf"([^.!?]*{term}[^.!?]*\s+is\s+[^.!?]+)",
            rf"Definition:\s*{term}\s*[-:]\s*([^.!?]+)",
            rf"{term}\s*\(\s*([^)]+)\s*\)",
        ]

        for pattern in definition_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                definition_text = match.group(1) if match.groups() else context
                return DefinitionDetection(
                    term=term,
                    has_definition=True,
                    definition_text=definition_text.strip(),
                    confidence=0.8,
                    context_sentence=context,
                )

        return DefinitionDetection(
            term=term, has_definition=False, confidence=0.3, context_sentence=context
        )

    def _extract_definition_from_context(
        self, term: str, context: str
    ) -> Optional[str]:
        patterns = [
            rf"{re.escape(term)}\s+is\s+([^.!?]+)",
            rf"{re.escape(term)}\s+means\s+([^.!?]+)",
            rf"{re.escape(term)}\s+refers\s+to\s+([^.!?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _calculate_semantic_similarity(self, term1: str, term2: str) -> float:
        if self.semantic_model is None:
            return 0.0

        try:
            emb1 = self.semantic_model.encode(term1, convert_to_numpy=True)
            emb2 = self.semantic_model.encode(term2, convert_to_numpy=True)

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)

        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def validate_term(
        self, term: str, context: Optional[str] = None
    ) -> ValidationResult:
        term_clean = term.strip()
        term_normalized = term_clean.lower()

        is_defined = term_normalized in self._term_index

        classification = self._classify_term_with_transformer(term_clean, context)

        definition_detection = None
        if context and not is_defined:
            definition_detection = self._detect_definition_with_transformer(
                term_clean, context
            )
            if definition_detection.has_definition:
                is_defined = True

        semantic_similarity = 0.0
        if not is_defined and self.semantic_model is not None:
            max_similarity = 0.0
            for defined_term in self.glossary.keys():
                similarity = self._calculate_semantic_similarity(
                    term_clean, defined_term
                )
                max_similarity = max(max_similarity, similarity)
            semantic_similarity = max_similarity

        suggested_definition = None
        if is_defined:
            preferred_form = self._term_index.get(term_normalized, term_clean)
            suggested_definition = self.glossary.get(preferred_form)
        elif definition_detection and definition_detection.definition_text:
            suggested_definition = definition_detection.definition_text

        return ValidationResult(
            term=term_clean,
            normalized_term=term_normalized,
            is_defined=is_defined,
            classification=classification,
            definition_detection=definition_detection,
            semantic_similarity_to_defined=semantic_similarity,
            suggested_definition=suggested_definition,
        )

    def _extract_sentence_context(self, text: str, term: str) -> Optional[str]:
        """Extract sentence context around a term with surrounding sentences"""
        try:
            # Split text into sentences using spaCy if available, otherwise regex
            if self.nlp:
                doc = self.nlp(text)
                sentences = [
                    sent.text.strip() for sent in doc.sents if sent.text.strip()
                ]
            else:
                # Fallback to regex sentence splitting
                sentences = re.split(r"[.!?]+", text)
                sentences = [s.strip() for s in sentences if s.strip()]

            # Find sentence containing the term
            for i, sentence in enumerate(sentences):
                if re.search(r"\b" + re.escape(term) + r"\b", sentence, re.IGNORECASE):
                    # Build context with previous and next sentences
                    context_parts = []

                    # Add previous sentence if available
                    if i > 0:
                        prev_sentence = sentences[i - 1]
                        # Take last part of previous sentence
                        if len(prev_sentence) > 100:
                            prev_sentence = "..." + prev_sentence[-100:]
                        context_parts.append(prev_sentence)

                    # Add current sentence
                    context_parts.append(sentence)

                    # Add next sentence if available
                    if i < len(sentences) - 1:
                        next_sentence = sentences[i + 1]
                        # Take first part of next sentence
                        if len(next_sentence) > 100:
                            next_sentence = next_sentence[:100] + "..."
                        context_parts.append(next_sentence)

                    # Join with proper spacing and punctuation
                    context = " ".join(context_parts)

                    # Limit total length
                    if len(context) > 300:
                        context = context[:300] + "..."

                    return context

            return None

        except Exception as e:
            print(f"Error extracting sentence context for '{term}': {e}")
            return None

    def extract_terms_from_text(self, text: str) -> List[str]:
        terms = set()

        if self.nlp:
            try:
                doc = self.nlp(text)
                for chunk in doc.noun_chunks:
                    if len(chunk.text.strip()) > 2 and len(chunk.text.strip()) <= 25:
                        terms.add(chunk.text.strip())

                for ent in doc.ents:
                    if len(ent.text.strip()) > 2 and len(ent.text.strip()) <= 25:
                        terms.add(ent.text.strip())

                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
                        start = max(0, token.i - 2)
                        end = min(len(doc), token.i + 3)
                        phrase = " ".join(
                            [
                                t.text
                                for t in doc[start:end]
                                if t.pos_ in ["NOUN", "PROPN", "ADJ", "NUM"]
                            ]
                        )
                        if len(phrase.split()) >= 2 and len(phrase) <= 25:
                            terms.add(phrase)
            except Exception as e:
                print(f"Error in spaCy term extraction: {e}")

        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        terms.update(acronyms)

        camelcase = re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", text)
        terms.update(camelcase)

        tech_suffixes = [
            "tion",
            "ment",
            "ness",
            "ity",
            "er",
            "or",
            "ism",
            "ist",
            "ive",
            "able",
            "ible",
            "al",
            "ial",
            "ic",
            "ical",
            "ous",
            "ious",
            "ful",
            "less",
            "ize",
            "ise",
            "fy",
            "ate",
            "en",
            "ify",
        ]
        for suffix in tech_suffixes:
            pattern = rf"\b\w+{suffix}\b"
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update([match for match in matches if 3 < len(match) <= 20])

        quoted = re.findall(r'"([^"]+)"', text)
        terms.update([quote for quote in quoted if 2 < len(quote) <= 25])

        common_english = {
            "the",
            "and",
            "have",
            "will",
            "time",
            "person",
            "way",
            "day",
            "man",
            "thing",
            "use",
            "make",
            "work",
            "part",
            "take",
            "get",
            "place",
            "live",
            "back",
            "only",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "a",
            "an",
        }

        filtered_terms = []
        for term in terms:
            term_lower = term.lower()
            if (
                term_lower not in common_english
                and len(term.strip()) > 2
                and len(term.strip()) <= 25
                and not term.isdigit()
                and not re.match(r"^[.,!?;:]+$", term)
            ):
                filtered_terms.append(term.strip())

        return sorted(list(set(filtered_terms)), key=len, reverse=True)

    def validate_terms(
        self,
        terms: List[str],
        contexts: Optional[List[str]] = None,
        full_text: Optional[str] = None,
    ) -> ValidationReport:
        if contexts is None:
            contexts = [None] * len(terms)

        valid_contexts = [ctx for ctx in contexts if ctx is not None]
        all_text = (
            " ".join(valid_contexts)
            if valid_contexts
            else (full_text or " ".join(terms))
        )
        extracted_definitions = self._extract_definitions_from_text(all_text)

        for term, definition in extracted_definitions.items():
            self.glossary[term] = definition

        self._term_index = self._build_term_index()

        defined_terms = []
        undefined_terms = []
        common_english_terms = []
        potential_domain_terms = []

        for term, context in zip(terms, contexts):
            result = self.validate_term(term, context)

            # Extract sentence context if full_text is available
            if full_text:
                result.context_excerpt = self._extract_sentence_context(full_text, term)

            if result.is_defined:
                defined_terms.append(result)
            elif result.classification.is_common_english:
                common_english_terms.append(result)
            elif result.classification.is_domain_term:
                undefined_terms.append(result)
                potential_domain_terms.append(result)
            else:
                undefined_terms.append(result)

        semantic_clusters = self._create_semantic_clusters(undefined_terms)

        statistics = {
            "total_terms": len(terms),
            "defined_terms": len(defined_terms),
            "undefined_terms": len(undefined_terms),
            "common_english_terms": len(common_english_terms),
            "potential_domain_terms": len(potential_domain_terms),
            "definition_coverage": len(defined_terms) / len(terms) if terms else 0,
            "avg_confidence": (
                np.mean(
                    [
                        r.classification.confidence
                        for r in defined_terms + undefined_terms
                    ]
                )
                if terms
                else 0
            ),
            "semantic_clusters_found": (
                len(semantic_clusters) if semantic_clusters else 0
            ),
        }

        return ValidationReport(
            defined_terms=defined_terms,
            undefined_terms=undefined_terms,
            common_english_terms=common_english_terms,
            potential_domain_terms=potential_domain_terms,
            statistics=statistics,
            semantic_clusters=semantic_clusters,
        )

    def _create_semantic_clusters(
        self, undefined_terms: List[ValidationResult]
    ) -> Dict[str, List[str]]:
        if self.semantic_model is None or not undefined_terms:
            return {}

        try:
            terms = [result.term for result in undefined_terms]
            embeddings = self.semantic_model.encode(terms, convert_to_numpy=True)

            clusters = defaultdict(list)
            used_indices = set()

            for i, term in enumerate(terms):
                if i in used_indices:
                    continue

                cluster_id = f"cluster_{len(clusters)}"
                clusters[cluster_id].append(term)
                used_indices.add(i)

                for j, other_term in enumerate(terms):
                    if i != j and j not in used_indices:
                        similarity = np.dot(embeddings[i], embeddings[j])
                        if similarity > 0.8:  # High similarity threshold
                            clusters[cluster_id].append(other_term)
                            used_indices.add(j)

            return dict(clusters)

        except Exception as e:
            print(f"Error creating semantic clusters: {e}")
            return {}

    def get_priority_undefined_terms(
        self, report: ValidationReport, top_k: int = 10
    ) -> List[Dict]:
        priority_terms = []

        for result in report.undefined_terms:
            if result.classification.is_domain_term:
                priority_score = result.classification.confidence

                if result.semantic_similarity_to_defined > 0.7:
                    priority_score += 0.2

                if (
                    result.definition_detection
                    and result.definition_detection.has_definition
                ):
                    priority_score += 0.1

                priority_terms.append(
                    {
                        "term": result.term,
                        "normalized_term": result.normalized_term,
                        "priority_score": priority_score,
                        "confidence": result.classification.confidence,
                        "semantic_similarity": result.semantic_similarity_to_defined,
                        "context": (
                            result.definition_detection.context_sentence
                            if result.definition_detection
                            else None
                        ),
                        "classification_reason": result.classification.classification_reason,
                        "suggested_definition": result.suggested_definition,
                    }
                )

        priority_terms.sort(key=lambda x: x["priority_score"], reverse=True)

        return priority_terms[:top_k]

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import string


@dataclass
class TermVariant:

    text: str
    is_acronym: bool = False
    is_abbreviation: bool = False
    is_plural: bool = False
    is_possessive: bool = False
    normalized_form: Optional[str] = None

    def __post_init__(self):
        if self.normalized_form is None:
            self.normalized_form = self.text.lower()


@dataclass
class TermDefinition:

    preferred_form: str
    definition: str
    variants: List[TermVariant]
    source: Optional[str] = None
    category: Optional[str] = None

    def __post_init__(self):
        has_preferred = any(
            v.text.lower() == self.preferred_form.lower() for v in self.variants
        )
        if not has_preferred:
            self.variants.append(TermVariant(text=self.preferred_form))

        for variant in self.variants:
            if variant.normalized_form is None:
                variant.normalized_form = variant.text.lower()

    def get_all_forms(self) -> List[str]:
        return [v.text for v in self.variants] + [self.preferred_form]

    def get_normalized_forms(self) -> List[str]:
        return [v.normalized_form for v in self.variants] + [
            self.preferred_form.lower()
        ]


@dataclass
class TermOccurrence:

    term: str
    normalized_term: str
    start_pos: int
    end_pos: int
    sentence: str
    sentence_position: int
    is_defined: bool = False
    is_definition: bool = False
    context: Optional[Dict] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class TermAnalysis:

    defined_terms: Dict[str, TermDefinition]
    term_occurrences: List[TermOccurrence]
    undefined_terms: List[TermOccurrence]
    potential_terms: List[TermOccurrence]
    statistics: Dict[str, int]

    def get_term_frequency(self) -> Dict[str, int]:
        freq = defaultdict(int)
        for occ in self.term_occurrences:
            freq[occ.normalized_term.lower()] += 1
        return dict(freq)

    def get_undefined_terms(self) -> List[Tuple[str, int]]:
        freq = defaultdict(int)
        for occ in self.undefined_terms:
            freq[occ.term] += 1
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)


class TerminologyAnalyzer:

    def __init__(self, glossary: Optional[Dict[str, TermDefinition]] = None):
        self.glossary = glossary or {}
        self._term_index = self._build_term_index()
        self._stopwords = self._load_stopwords()
        self._acronym_pattern = re.compile(r"\b[A-Z]{2,}s?\b")
        self._camel_case_pattern = re.compile(r"([A-Z][a-z]+)")

    def _build_term_index(self) -> Dict[str, str]:
        index = {}
        for term_def in self.glossary.values():
            for variant in term_def.variants:
                index[variant.normalized_form] = term_def.preferred_form
            index[term_def.preferred_form.lower()] = term_def.preferred_form
        return index

    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords"""
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "at",
            "from",
            "by",
            "on",
            "off",
            "for",
            "in",
            "out",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "couldn",
            "didn",
            "doesn",
            "hadn",
            "hasn",
            "haven",
            "isn",
            "ma",
            "mightn",
            "mustn",
            "needn",
            "shan",
            "shouldn",
            "wasn",
            "weren",
            "won",
            "wouldn",
        }

    def add_glossary_entries(self, entries: List[TermDefinition]) -> None:
        for entry in entries:
            self.glossary[entry.preferred_form] = entry
        self._term_index = self._build_term_index()

    def analyze_document(self, text: str) -> TermAnalysis:
        sentences = self._split_into_sentences(text)

        term_occurrences = []
        undefined_terms = []
        potential_terms = []

        full_text_start = 0
        for sent_idx, sentence in enumerate(sentences):
            candidates = self._extract_candidate_terms(sentence)
            sentence_start = text.find(sentence, full_text_start)
            sentence_end = sentence_start + len(sentence)

            for candidate in candidates:
                normalized = candidate.lower()

                if normalized in self._term_index:
                    preferred_form = self._term_index[normalized]
                    term_def = self.glossary[preferred_form]

                    candidate_start = sentence.find(candidate)
                    occurrence = TermOccurrence(
                        term=candidate,
                        normalized_term=preferred_form,
                        start_pos=sentence_start + candidate_start,
                        end_pos=sentence_start + candidate_start + len(candidate),
                        sentence=sentence,
                        sentence_position=sent_idx,
                        is_defined=True,
                        context={
                            "preferred_form": preferred_form,
                            "definition": term_def.definition,
                        },
                    )
                    term_occurrences.append(occurrence)
                else:
                    if self._is_potential_term(candidate):
                        candidate_start = sentence.find(candidate)
                        occurrence = TermOccurrence(
                            term=candidate,
                            normalized_term=normalized,
                            start_pos=sentence_start + candidate_start,
                            end_pos=sentence_start + candidate_start + len(candidate),
                            sentence=sentence,
                            sentence_position=sent_idx,
                            is_defined=False,
                        )
                        term_occurrences.append(occurrence)
                        undefined_terms.append(occurrence)
                        potential_terms.append(occurrence)

            full_text_start = sentence_end

        defined_terms = self._extract_definitions(text)

        for term_def in defined_terms.values():
            if term_def.preferred_form not in self.glossary:
                self.glossary[term_def.preferred_form] = term_def
                self._term_index = self._build_term_index()

        stats = {
            "total_terms": len(term_occurrences),
            "defined_terms": len([t for t in term_occurrences if t.is_defined]),
            "undefined_terms": len(undefined_terms),
            "potential_terms": len(potential_terms),
            "new_definitions": len(defined_terms),
        }

        return TermAnalysis(
            defined_terms=dict(self.glossary),
            term_occurrences=term_occurrences,
            undefined_terms=undefined_terms,
            potential_terms=potential_terms,
            statistics=stats,
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        # TODO: simplify with spacy
        text = text.replace("\n", " ")

        sentence_enders = re.compile(r"[.!?]\s+")
        sentences = []
        start = 0

        for match in sentence_enders.finditer(text):
            end = match.end() - 1  # Back up to the period
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end + 1

        last_sentence = text[start:].strip()
        if last_sentence:
            sentences.append(last_sentence)

        return sentences

    def _extract_candidate_terms(self, text: str) -> List[str]:
        candidates = set()

        parenthetical_content = []
        for match in re.finditer(r"\(([^)]+)\)", text):
            content = match.group(1)
            parenthetical_content.append(content)

            if re.match(r"^[A-Z]{2,}$", content.strip()):
                candidates.add(content.strip())

            words_in_parens = content.strip().split()
            if len(words_in_parens) <= 3:  # Reasonable length for terms
                candidates.add(content.strip())

        clean_text = re.sub(r"\([^)]*\)", "", text)
        words = clean_text.split()

        for i, word in enumerate(words):
            word_clean = word.strip(string.punctuation)
            if self._is_potential_term(word_clean):
                candidates.add(word_clean)

        for i in range(len(words)):
            for length in range(2, min(4, len(words) - i + 1)):
                phrase_words = words[i : i + length]

                if (
                    phrase_words[0].lower() in self._stopwords
                    or phrase_words[-1].lower() in self._stopwords
                ):
                    continue

                phrase = " ".join(phrase_words)
                phrase = phrase.strip(string.punctuation)

                if self._is_potential_term(phrase):
                    candidates.add(phrase)

        for match in self._acronym_pattern.finditer(text):
            candidates.add(match.group(0))

        return list(candidates)

    def _is_potential_term(self, text: str) -> bool:
        """Determine if a string could be a term"""
        if not text or len(text) < 2:
            return False

        if text.replace(".", "").isdigit():
            return False

        if len(text) == 1 and not text.isupper():
            return False

        if text.lower() in self._stopwords:
            return False

        if all(c in string.punctuation for c in text):
            return False

        if len(text) <= 3 and text.lower() in [
            "is",
            "an",
            "the",
            "and",
            "or",
            "but",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
        ]:
            return False

        words = text.split()
        if len(words) > 1 and words[0].lower() in [
            "is",
            "an",
            "the",
            "and",
            "or",
            "but",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
        ]:
            return False

        if len(words) > 1 and words[-1].lower() in [
            "is",
            "an",
            "the",
            "and",
            "or",
            "but",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
        ]:
            return False

        return True

    def _is_likely_term(self, text: str) -> bool:
        """Determine if a string is likely to be a domain term"""
        if " " in text:
            return True

        if text.isupper() and len(text) >= 2:
            return True

        if text.istitle() and len(text) > 3:
            return True

        if self._camel_case_pattern.search(text):
            return True

        return False

    def _extract_definitions(self, text: str) -> Dict[str, TermDefinition]:
        definitions = {}

        patterns = [
            r"([A-Z][a-zA-Z0-9\s\-.]+?)\s+(?:is|means|refers to|denotes)\s+([A-Z][^.!?]+?)(?=[.!?]|$)",
            r"([A-Z][a-zA-Z0-9\s\-.]+?)\s*[\-:]\s*([^\n]+?)(?=[.!?]|$)",
            r"\(([A-Z][a-zA-Z0-9\s\-.]+?)\)\s*[\-:]?\s*([^\n]+?)(?=[.!?]|$)",
            r"([A-Z]{2,})\s*[\-:]?\s*([^\n]+?)(?=[.!?]|$)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                term = match.group(1).strip()
                definition = match.group(2).strip()

                term = term.strip("\"'()[]{}")
                definition = definition.strip("\"'()[]{}")

                if term and definition and len(term.split()) <= 5:
                    term_def = TermDefinition(
                        preferred_form=term,
                        definition=definition,
                        variants=[],
                        source="extracted",
                    )
                    definitions[term.lower()] = term_def

        return definitions

    def suggest_term_variants(self, term: str) -> List[TermVariant]:
        variants = []

        variants.append(TermVariant(text=term, normalized_form=term.lower()))

        if term.endswith("s") and not term.endswith("ss"):
            singular = term[:-1]
            variants.append(
                TermVariant(
                    text=singular, normalized_form=singular.lower(), is_plural=False
                )
            )
        else:
            plural = term + "s"
            variants.append(
                TermVariant(text=plural, normalized_form=plural.lower(), is_plural=True)
            )

        if not term.endswith("'s"):
            possessive = term + "'s"
            variants.append(
                TermVariant(
                    text=possessive,
                    normalized_form=term.lower(),  # Normalize by removing 's for comparison
                    is_possessive=True,
                )
            )

        if " " in term and len(term) > 2:
            acronym = "".join(word[0].upper() for word in term.split() if word)
            if len(acronym) >= 2:  # Only consider if at least 2 letters
                variants.append(
                    TermVariant(
                        text=acronym, normalized_form=term.lower(), is_acronym=True
                    )
                )

        return variants

    def find_similar_terms(
        self, term: str, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:

        def similarity(a: str, b: str) -> float:
            a = a.lower()
            b = b.lower()

            if a == b:
                return 1.0

            if a in b or b in a:
                return 0.8

            a_words = set(a.split())
            b_words = set(b.split())
            common = a_words & b_words
            union = a_words | b_words

            if not union:
                return 0.0

            return len(common) / len(union)

        similarities = []
        for known_term in self.glossary.keys():
            score = similarity(term, known_term)
            if score >= threshold:
                similarities.append((known_term, score))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

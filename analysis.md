# Modular NLP/NLI Analysis Package

## Diagram 1: Document Processor
### Segmentation & Classification

```mermaid
flowchart TD
    RAW["Raw text string"]

    SPLIT["Split into lines\nMerge continuation lines\nIdentify headings via regex"]


    SPACY["SpaCy POS + dep parse\nper line"]

    MODAL_STRONG{"Strong modal?\nshall · must"}
    MODAL_WEAK{"Weak modal?\nshould · may · can"}
    LEMMA{"Informative lemma?\nexample · note · describe"}
    VERB{"Normative verb?\nrequire · prohibit · mandate"}
    IMPERATIVE{"Syntactic\nimperative?\n(no subject)"}

    NORM["NORMATIVE"]
    INFO["INFORMATIVE"]
    UNK["UNKNOWN"]

    CONF["_calculate_confidence\nNORMATIVE: 0.45 – 0.95\nINFORMATIVE: 0.60 – 0.90\nUNKNOWN: 0.40"]

    ANCHORS["extract_semantic_anchors\nSemanticAnchor objects\nper segment"]

    OUT(["ProcessedDocument\nsegments\nprocessing_stats"])

    RAW --> SPLIT --> SPACY
    SPACY --> LEMMA
    LEMMA -->|yes| INFO
    LEMMA -->|no| MODAL_STRONG
    MODAL_STRONG -->|yes| NORM
    MODAL_STRONG -->|no| MODAL_WEAK
    MODAL_WEAK -->|yes| INFO
    MODAL_WEAK -->|no| VERB
    VERB -->|yes| NORM
    VERB -->|no| IMPERATIVE
    IMPERATIVE -->|yes| NORM
    IMPERATIVE -->|no| UNK
    NORM & INFO & UNK --> CONF --> ANCHORS --> OUT
```



## Diagram 2a: Term Extraction Methods

```mermaid
flowchart TD
    IN["Full document text\n+ segments"]

    subgraph SPACY_METHODS["SpaCy-Based Extraction"]
        E1["Noun chunks\n3–25 chars"]
        E2["Named entities\n2–25 chars"]
        E3["Noun phrase construction\n2+ token phrases"]
    end

    FILTER["Score & deduplicate\nkeep high-quality candidates"]

    OUT(["Candidate term list"])

    IN --> SPACY_METHODS
    E1 & E2 & E3 --> FILTER
    FILTER --> OUT
```
## Diagram 2b: Term Classification

```mermaid
flowchart TD
    IN(["Candidate term list"])

    subgraph TRANSFORMER["Transformer Path"]
        T1["RoBERTa tokenizer"]
        T2["RoBERTa model"]
        T3{"confidence > 0.7?"}
        T1 --> T2 --> T3
    end

    subgraph RULES["Rule-Based Fallback"]
        R1{"Acronym</br>or CamelCase?"}
        R2{"Multi-word</br>phrase?"}
        R3{"Technical</br>suffix?"}
    end

    DOMAIN["Domain</br>term\n(keep)"]
    COMMON["Common</br>English\n(discard)"]

    OUT(["Validated domain terms"])

    IN --> TRANSFORMER
    T3 -->|yes| DOMAIN
    T3 -->|no, use fallback| RULES
    R1 & R2 & R3 -->|yes| DOMAIN
    R1 & R2 & R3 -->|no to all| COMMON
    DOMAIN --> OUT
```
## Diagram 2c: Definition Detection

```mermaid
flowchart TD
    IN(["Validated domain terms\n+ full document text"])

    subgraph REGEX["Regex Pattern Matching"]
        D1["X is Y"]
        D2["X means Y"]
        D3["X refers to Y"]
        D4["X: definition"]
        D5["X: definition"]
    end

    subgraph ML["Transformer Verification"]
        ML1["RoBERTa tokenizer + model"]
        ML2{"Is term X defined\nin this context?"}
    end

    GLOSSARY["Glossary index\nterm → definition text"]

    DEFINED(["defined_terms\nwith extracted definition text"])
    UNDEFINED(["undefined_terms\nno definition found"])

    IN --> REGEX & ML
    D1 & D2 & D3 & D4 & D5 --> GLOSSARY
    ML1 --> ML2
    ML2 -->|yes| GLOSSARY
    ML2 -->|no| UNDEFINED
    GLOSSARY --> DEFINED
```
## Diagram 2d: Semantic Clustering

```mermaid
flowchart TD
    IN(["undefined_terms"])

    ENC["SentenceTransformer\nall-MiniLM-L6-v2\nencode each term → vector"]

    MATRIX["Compute cosine similarity\nmatrix across all term pairs"]

    THRESH{"similarity > 0.8?"}

    GROUP["Group into cluster\nassign cluster representative"]
    SOLO["Term stays ungrouped"]

    OUT(["ValidationReport\ndefined_terms\nundefined_terms\nterm_clusters\nstatistics"])

    IN --> ENC --> MATRIX --> THRESH
    THRESH -->|yes| GROUP
    THRESH -->|no| SOLO
    GROUP & SOLO --> OUT
```
## Diagram 3a: Candidate Pair Generation

```mermaid
flowchart TD
    IN["All normative +\nunknown segments"]

    PAIRS["Generate all O(n²)\nstatement pairs"]

    OVERLAP{"Term overlap check\n≥ 2 shared non-stopwords\nin common?"}

    DIST{"Segment distance check\n|index_a − index_b|\n≤ max_segment_distance?"}

    SKIP1(["Discard pair\n(unrelated topics)"])
    SKIP2(["Discard pair\n(too far apart)"])

    OUT(["Candidate pair queue"])

    IN --> PAIRS --> OVERLAP
    OVERLAP -->|no| SKIP1
    OVERLAP -->|yes| DIST
    DIST -->|no| SKIP2
    DIST -->|yes| OUT
```
## Diagram 3b: Stage 1: Rule-Based Contradiction Detection

```mermaid
flowchart TD
    IN(["Candidate pair queue"])

    MODAL{"Modal contradiction?\nmust  vs  must not\nshall  vs  shall not\nrequired  vs  prohibited"}

    VALUE{"Value antonym?\ntrue ↔ false\nenabled ↔ disabled\nrequired ↔ optional\nallow ↔ deny"}

    QUANTITY{"Quantity contradiction?\nDifferent numeric values\n+ quantity operators\non the same subject"}

    FOUND["Rule-based Contradiction\nhigh confidence"]
    PASS["No rule match\npass to Stage 2"]

    OUT(["Contradiction\ntype · explanation\nconfidence: high"])

    IN --> MODAL
    MODAL -->|yes| FOUND
    MODAL -->|no| VALUE
    VALUE -->|yes| FOUND
    VALUE -->|no| QUANTITY
    QUANTITY -->|yes| FOUND
    QUANTITY -->|no| PASS
    FOUND --> OUT
```
## Diagram 3c: Stage 2: ML-Based Contradiction Detection (NLI)

```mermaid
flowchart TD
    IN(["Candidate pairs\nthat passed Stage 1"])

    LEN{"Both statements\n≥ 6 words?"}
    SKIP(["Skip: too short\nfor reliable NLI"])

    ENCODE["CrossEncoder\nnli-deberta-v3-base\npredict(statement_a, statement_b)"]

    SOFTMAX["Softmax over 3 labels\n[contradiction, entailment, neutral]"]

    LABEL{"Top label =\ncontradiction?"}

    SCORE{"Score >\nml_threshold?"}

    FOUND["ML Contradiction\nwith confidence score"]
    DISCARD(["Discard pair\n(entailment or neutral)"])

    OUT(["Contradiction\ntype · statement_a · statement_b\nconfidence · explanation"])

    IN --> LEN
    LEN -->|no| SKIP
    LEN -->|yes| ENCODE --> SOFTMAX --> LABEL
    LABEL -->|no| DISCARD
    LABEL -->|yes| SCORE
    SCORE -->|no| DISCARD
    SCORE -->|yes| FOUND --> OUT
```
## Diagram 4a: S2 Taxonomy: Routing Overview

```mermaid
flowchart TD
    IN["ProcessedDocument\nsegments"]

    SPLIT{"Segment\nelement_type"}

    NORM["NORMATIVE\nsegments"]
    INFO["INFORMATIVE\nsegments"]

    F002["FAIL-002\nSynonym Inconsistency"]
    F004["FAIL-004\nModal Inconsistency"]
    F006["FAIL-006\nVague Qualifiers"]
    F007["FAIL-007\nAmbiguous Referent"]
    F008["FAIL-008\nMissing Temporal Anchor"]
    F005["FAIL-005\nHidden Normative"]

    OUT(["S2Finding\nuid · error_type · category\nseverity · text_span · confidence"])

    IN --> SPLIT
    SPLIT --> NORM & INFO
    NORM --> F002 & F004 & F006 & F007 & F008
    INFO --> F005
    F002 & F004 & F005 & F006 & F007 & F008 --> OUT
```
## Diagram 4b: FAIL-002 and FAIL-004

```mermaid
flowchart TD
    IN(["NORMATIVE segments"])

    subgraph F002["FAIL-002: Synonym Inconsistency"]
        A1["Extract role nouns\nbefore modal verbs\nOperator · Technician\nUser · Admin · Client"]
        A2{"Two different role nouns\nfound in document?"}
        A3{"Both belong to the\nsame synonym group?"}
        A4(["Finding: same actor\nreferred to by\ninconsistent names"])
        A1 --> A2
        A2 -->|yes| A3
        A3 -->|yes| A4
    end

    subgraph F004["FAIL-004: Modal Inconsistency"]
        B1["Obligation pool\nshall · must"]
        B2["Permission pool\nmay · can"]
        B3{"Pair from each pool\nshares ≥ 2\nnon-stopwords?"}
        B4(["Finding: same topic\nhas both obligation\nAND permission"])
        B1 & B2 --> B3
        B3 -->|yes| B4
    end

    IN --> F002 & F004
```

## Diagram 4c: FAIL-005 and FAIL-006

```mermaid
flowchart TD
    NORM(["NORMATIVE segments"])
    INFO(["INFORMATIVE segments"])

    subgraph F005["FAIL-005: Hidden Normative"]
        A1["Scan informative\nsegments for\nnormative language"]
        A2{"Match found?\nshould · is expected to\nmust · shall · required"}
        A3(["Finding: normative\nrequirement hidden\nin informative section"])
        A1 --> A2 -->|yes| A3
    end

    subgraph F006["FAIL-006: Vague Qualifiers"]
        B1["Scan normative\nsegments only"]
        B2{"Match found?\ntoo · very · quite · rather\nsomewhat · approximately\nabout · generally · typically"}
        B3(["Finding: imprecise\nqualifier in a\nnormative statement"])
        B1 --> B2 -->|yes| B3
    end

    INFO --> F005
    NORM --> F006
```

## Diagram 4d: FAIL-007 and FAIL-008

```mermaid
flowchart TD
    IN(["NORMATIVE segments"])

    subgraph F007["FAIL-007: Ambiguous Referent"]
        A1["Scan for pronouns\nbefore modal verbs"]
        A2{"Match?\nit · this · that · they\n+ modal verb"}
        A3(["Finding: pronoun\nwith unclear referent"])
        A1 --> A2 -->|yes| A3
    end

    subgraph F008["FAIL-008: Missing Temporal Anchor"]
        B1["Scan normative segments"]

        B2{"Vague temporal word?\nASAP · promptly · soon\neventually · in due course"}

        B3{"Time-sensitive verb?\nnotify · report · submit\ndeliver · complete · provide"}

        B4{"Specific deadline\nsuppressor present?\nwithin N days\nby DATE · no later than"}

        B5(["Finding: vague\ntemporal: sub-case A"])
        B6(["Finding: deadline verb\nwith no anchor\n— sub-case B"])

        B1 --> B2 & B3
        B2 -->|yes| B5
        B3 -->|yes| B4
        B4 -->|no suppressor found| B6
        B4 -->|suppressor found| CLEAR(["No finding"])
    end

    IN --> F007 & F008
```
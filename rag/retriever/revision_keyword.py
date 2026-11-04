"""Pure request-local BM25. No provider, vector store, shared corpus or source execution."""

import math
import re
import unicodedata
from collections.abc import Callable, Iterable

from rank_bm25 import BM25Okapi

VERSION = "keyword-bm25-positive-idf-v1"
_WORDS = re.compile(r"\w+", re.UNICODE)
_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
MAX_TERM_CHARACTERS = 128


class CorpusLimitError(ValueError):
    pass


class ScoringError(ValueError):
    pass


def tokenize(text: str, budget: int) -> list[str]:
    """NFC words: whole identifiers plus snake/ASCII camel components; punctuation ignored."""
    tokens = []
    for match in _WORDS.finditer(unicodedata.normalize("NFC", text)):
        identifier = match.group().strip("_")
        # Camel regexes only operate on bounded identifiers, avoiding pathological long input.
        if not identifier or len(identifier) > MAX_TERM_CHARACTERS:
            continue
        parts = _CAMEL.sub(r"\1 \2", _ACRONYM.sub(r"\1 \2", identifier)).replace("_", " ").split()
        occurrence = dict.fromkeys([identifier.casefold(), *(part.casefold() for part in parts)])
        for value in occurrence:
            if value and len(value) <= MAX_TERM_CHARACTERS:
                if len(tokens) >= budget:
                    raise CorpusLimitError("Keyword token budget exceeded")
                tokens.append(value)
    return tokens


class _PositiveBM25(BM25Okapi):
    """Same Okapi TF/length formula, with positive log1p IDF, including tiny corpora."""

    def _calc_idf(self, nd):
        self.idf = {
            term: math.log1p((self.corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in nd.items()
        }
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


def order_scores(
    ids: list[str], documents: list[list[str]], terms: list[str], scores: Iterable[float]
) -> list[tuple[str, float]]:
    values = list(scores)
    if len(values) != len(ids):
        raise ScoringError("Invalid score cardinality")
    result = []
    query = set(terms)
    for ident, tokens, score in zip(ids, documents, values, strict=True):
        number = float(score)
        if not math.isfinite(number):
            raise ScoringError("Non-finite keyword score")
        if number > 0 and query.intersection(tokens):
            result.append((ident, number))
    return sorted(result, key=lambda item: (-item[1], item[0]))


def rank(
    corpus: list[tuple[str, str]],
    query: str,
    token_budget: int,
    stop: Callable[[], bool] = lambda: False,
) -> tuple[list[tuple[str, float]], int]:
    terms = list(dict.fromkeys(tokenize(query, 4096)))
    documents = []
    remaining = token_budget
    for _, value in corpus:
        if stop():
            raise TimeoutError("Search interrupted")
        tokens = tokenize(value, remaining)
        remaining -= len(tokens)
        documents.append(tokens)
    if not terms or not any(documents):
        return [], len(terms)
    if stop():
        raise TimeoutError("Search interrupted")
    model = _PositiveBM25(documents, k1=1.5, b=0.75)
    scores = model.get_scores(terms)
    return order_scores([ident for ident, _ in corpus], documents, terms, scores), len(terms)

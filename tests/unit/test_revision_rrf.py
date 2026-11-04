"""Deterministic RRF mathematics and strict hybrid request contracts."""

from copy import deepcopy
from fractions import Fraction
from uuid import UUID

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.schemas.search import HybridSearchRequest, KeywordMatch, SemanticMatch
from core.services.hybrid_retrieval import candidate_depth
from rag.retriever.revision_rrf import RRF_K, fuse


def match(ident=1, rank=1, semantic=False, **values):
    data = dict(
        source_id=UUID(int=100),
        chunk_id=UUID(int=ident),
        path="a.py",
        commit_sha="a" * 40,
        line_start=1,
        line_end=1,
        excerpt="canonical",
        rank=rank,
    )
    data.update(cosine_distance=0.5) if semantic else data.update(bm25_score=3.0)
    data.update(values)
    return (SemanticMatch if semantic else KeywordMatch)(**data)


@pytest.mark.parametrize(
    "kw,sem,expected",
    [
        ([], [], []),
        ([1], [], [1]),
        ([], [2], [2]),
        ([1], [1], [1]),
        ([1, 2], [2, 1], [1, 2]),
        ([1, 2], [3, 2], [2, 1, 3]),
        ([3, 2, 1], [1, 2, 3], [1, 3, 2]),
    ],
)
def test_exact_math_membership_and_order(kw, sem, expected):
    a = [match(i, n) for n, i in enumerate(kw, 1)]
    b = [match(i, n, True) for n, i in enumerate(sem, 1)]
    saved = deepcopy((a, b))
    result = fuse(a, b, 20)
    assert RRF_K == 60
    assert [m.chunk_id.int for m in result] == expected
    for n, row in enumerate(result, 1):
        terms = [
            Fraction(1, 60 + items.index(row.chunk_id.int) + 1)
            for items in (kw, sem)
            if row.chunk_id.int in items
        ]
        assert row.rrf_score == float(sum(terms)) and row.rank == n
        assert row.retrieval_source == (
            "both" if len(terms) == 2 else "keyword" if row.chunk_id.int in kw else "semantic"
        )
    assert (a, b) == saved
    assert fuse(a, b, 20) == result


def test_scores_do_not_mix_and_duplicate_contributes_once():
    a = [match(1, 2, bm25_score=1e100), match(1, 1, bm25_score=1e-10)]
    b = [match(2, 1, True, cosine_distance=2)]
    result = fuse(a, b, 20)
    assert [m.chunk_id.int for m in result] == [1, 2]
    assert result[0].rrf_score == result[1].rrf_score == 1 / 61
    assert result[0].keyword_rank == 1 and result[0].bm25_score == 1e-10
    assert result[0].cosine_distance is None and result[1].bm25_score is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 0),
        ("rank", -1),
        ("rank", True),
        ("rank", float("nan")),
        ("rank", 1.5),
        ("bm25_score", float("inf")),
        ("bm25_score", float("nan")),
        ("bm25_score", 0),
        ("bm25_score", True),
    ],
)
def test_invalid_keyword_metadata(field, value):
    row = match().model_copy(update={field: value})
    with pytest.raises(ValueError):
        fuse([row], [], 5)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, 3, True])
def test_invalid_semantic_scores(value):
    row = match(semantic=True).model_copy(update={"cosine_distance": value})
    with pytest.raises(ValueError):
        fuse([], [row], 5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "other.py"),
        ("excerpt", "different"),
        ("line_end", 2),
        ("commit_sha", "b" * 40),
        ("source_id", UUID(int=101)),
    ],
)
def test_citation_disagreement(field, value):
    with pytest.raises(ValueError, match="provenance"):
        fuse([match()], [match(semantic=True, **{field: value})], 5)


@pytest.mark.parametrize("top_k", [0, 21, True, 1.5])
def test_candidate_and_topk_limits(top_k):
    with pytest.raises(HTTPException):
        candidate_depth(top_k)
    with pytest.raises(ValueError):
        fuse([], [], top_k)


def test_candidate_depth_is_not_requested_topk():
    assert candidate_depth(1) == candidate_depth(5) == candidate_depth(20) == 20
    with pytest.raises(ValueError):
        fuse([match(i, i) for i in range(1, 22)], [], 5)
    assert len(fuse([match(1), match(2, 2)], [], 1)) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"rrf_constant": 1},
        {"owner_id": "foreign"},
        {"model": "arbitrary"},
        {"configuration": "http://attacker"},
        {"mode": "semantic"},
        {"top_k": 21},
        {"index_id": "bad"},
    ],
)
def test_hybrid_requests_forbid_untrusted_controls(extra):
    with pytest.raises(ValidationError):
        HybridSearchRequest.model_validate({"mode": "hybrid", "query": "JWT", **extra})


@pytest.mark.parametrize("top_k,expected", [(5, 5), (6, None)])
def test_lower_configured_candidate_cap(monkeypatch, top_k, expected):
    from core.config import settings

    monkeypatch.setattr(settings, "keyword_search_max_top_k", 5)
    if expected is None:
        with pytest.raises(HTTPException) as exc:
            candidate_depth(top_k)
        assert exc.value.status_code == 422
    else:
        assert candidate_depth(top_k) == expected

"""Deterministic, provider-free keyword ranking and transport contracts."""

import math

import pytest
from pydantic import ValidationError

from api.schemas.search import KeywordSearchRequest
from rag.retriever.revision_keyword import (
    CorpusLimitError,
    ScoringError,
    order_scores,
    rank,
    tokenize,
)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("get_user", {"get_user", "get", "user"}),
        ("getUser", {"getuser", "get", "user"}),
        ("HTTPServer", {"httpserver", "http", "server"}),
        ("CAFÉ", {"café"}),
        ("cafe\u0301", {"café"}),
        ("a += b.foo(x)", {"a", "b", "foo", "x"}),
        ("___", set()),
        ("!" * 100, set()),
        ("a" * 129, set()),
    ],
)
def test_token_policy(word, expected):
    assert set(tokenize(word, 100)) == expected


@pytest.mark.parametrize("query", ["user", "USER", "user user user"])
def test_real_positive_scores_in_single_and_common_term_corpora(query):
    result, count = rank([("b", "user"), ("a", "user"), ("c", "")], query, 100)
    assert [row[0] for row in result] == ["a", "b"]
    assert count == 1
    assert all(math.isfinite(score) and score > 0 for _, score in result)
    assert rank([("a", "user")], query, 100)[0][0][1] > 0


@pytest.mark.parametrize(
    "corpus,query",
    [
        ([], "x"),
        ([("a", "")], "x"),
        ([("a", "foo")], "!!!"),
        ([("a", "foo")], "bar"),
    ],
)
def test_empty_and_nonmatching(corpus, query):
    assert rank(corpus, query, 100)[0] == []


def test_relevance_and_document_frequency_are_real():
    result, _ = rank(
        [("b", "user user user"), ("a", "user dog cat"), ("c", "dog cat")], "user", 100
    )
    assert [x[0] for x in result] == ["b", "a"]
    assert result[0][1] > result[1][1]
    assert rank([("a", "getUser")], "get_user", 100)[0]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fails_closed(score):
    with pytest.raises(ScoringError):
        order_scores(["a"], [["x"]], ["x"], [score])


@pytest.mark.parametrize("score", [0, -1])
def test_nonpositive_excluded(score):
    assert order_scores(["a"], [["x"]], ["x"], [score]) == []


def test_fabricated_positive_nonmatch_excluded():
    assert order_scores(["a"], [["other"]], ["x"], [1]) == []
    with pytest.raises(ScoringError):
        order_scores(["a"], [["x"]], ["x"], [])


def test_token_budget_and_cooperative_stop():
    with pytest.raises(CorpusLimitError):
        rank([("a", "foo bar")], "foo", 1)
    with pytest.raises(TimeoutError):
        rank([("a", "foo")], "foo", 100, lambda: True)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"query": ""},
        {"query": "  "},
        {"query": 1},
        {"query": "x", "top_k": True},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 21},
        {"query": "x", "mode": "hybrid"},
        {"query": "x", "profile_id": "foreign"},
        {"query": "x" * 1025},
    ],
)
def test_typed_request_rejects_invalid(data):
    with pytest.raises(ValidationError):
        KeywordSearchRequest.model_validate(data)

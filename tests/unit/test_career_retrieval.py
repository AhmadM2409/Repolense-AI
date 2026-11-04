"""Offline lexical/window policy controls; no frozen quality fixture or model call."""

from types import SimpleNamespace

import pytest

from core.services import career_retrieval as retrieval


def requirement(skill="python", quote="Read CSV records using Python."):
    return {"skill": skill, "aliases": [], "quote": quote, "depth": quote}


def candidate(text, path="reader.py"):
    return SimpleNamespace(excerpt=text, path=path)


def test_query_preserves_object_action_and_deduplicates():
    query = retrieval.query(requirement())
    assert {"read", "csv", "records", "python"} <= query.terms
    assert "csv" in query.anchors and "python" not in query.anchors
    assert query.text.split().count("csv") == 1
    assert "required" not in retrieval.query(requirement(quote="Python required.")).terms


def test_query_bound_keeps_whole_terms_and_falls_back_to_skill():
    query = retrieval.query(requirement(quote=" ".join("token" + str(i) for i in range(60))))
    assert 0 < len(query.text) <= 256 and query.text.split()[-1].startswith("token")
    assert retrieval.query(requirement(skill="function", quote="Function required.")).text


def test_window_skips_oversized_header_and_preserves_whole_callable():
    lines = ["# reader", "# " + "x" * 400, "", "def read_csv(rows):", "    return list(rows)"]
    found = retrieval.window(candidate("\n".join(lines)), retrieval.query(requirement()))
    score, start, end, text = found
    assert score[1] and start >= 2
    assert text == "\n".join(lines[start : end + 1]) and "def read_csv" in text
    assert len(text.encode()) <= 300 and end - start + 1 <= 8


@pytest.mark.parametrize(
    "path,text",
    [
        ("styles.css", ".python { display: block; }"),
        ("reader.js", "function read_csv(rows) { return rows; }"),
        ("README.md", "# Python\n" + "CSV " * 110),
        ("reader.py", "def unrelated():\n    return True"),
    ],
)
def test_irrelevant_language_only_and_unretainable_hits_are_not_evidence(path, text):
    assert retrieval.window(candidate(text, path), retrieval.query(requirement())) is None


def test_requested_documentation_is_eligible():
    query = retrieval.query(requirement("sql", "SQL documentation preferred."))
    found = retrieval.window(candidate("# SQL\nSQL support is documented.", "README.md"), query)
    assert query.documentation and found is not None and found[0][0]


@pytest.mark.parametrize(
    "skill,path,other",
    [
        ("rust", "reader.rs", "reader.py"),
        ("typescript", "reader.ts", "reader.js"),
        ("c#", "reader.cs", "reader.c"),
    ],
)
def test_explicit_language_scope_is_not_tokenized_as_another_language(skill, path, other):
    query = retrieval.query(requirement(skill, "Read CSV records."))
    assert retrieval.window(candidate("// CSV records\nreturn csv_records;", path), query)
    assert retrieval.window(candidate("// CSV records\nreturn csv_records;", other), query) is None


def test_adjacent_braces_cannot_make_partial_routine_complete():
    text = "return old;\n}\n// Read CSV\nfunc read_csv() {\n  consume();"
    found = retrieval.window(candidate(text, "reader.go"), retrieval.query(requirement("go")))
    assert found and not found[0][1]


def test_multibyte_limit_is_bytes_and_does_not_slice_source_lines():
    lines = ["# " + "é" * 160, "def read_csv(rows):", "    return rows"]
    found = retrieval.window(candidate("\n".join(lines)), retrieval.query(requirement()))
    assert found and found[1] == 1 and found[3] == "\n".join(lines[1:])
    assert len(found[3].encode()) <= 300


def test_implementation_is_preferred_but_source_support_is_not_invented():
    query = retrieval.query(requirement())
    code = retrieval.window(candidate("def read_csv(rows):\n    return rows"), query)
    docs = retrieval.window(candidate("Read CSV records using Python.", "README.md"), query)
    assert code[0][:4] > docs[0][:4]

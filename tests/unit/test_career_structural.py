"""New independent source/terminology controls; no models, vectors or source execution."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.contracts import TextChunk
from core.services import career_context, career_evidence, career_retrieval, career_structural

FLASK = (
    "from flask import Blueprint, request, abort\n"
    'bp = Blueprint("orders", __name__)\n'
    '@bp.post("/orders")\n'
    "def store():\n"
    "    entry = request.get_json()\n"
    '    if not entry.get("sku"):\n'
    "        abort(422)\n"
    "    return entry\n"
)
GO = (
    'package edge\nimport "net/http"\n'
    "func receive(w http.ResponseWriter, r *http.Request) {\n"
    "    if r.Body == nil {\n"
    '        http.Error(w, "body needed", 422)\n'
    "        return\n"
    "    }\n"
    "    w.WriteHeader(202)\n"
    "}\n"
)


def snapshot(files):
    chunks = []
    for path, text in files.items():
        chunks.append(
            (
                TextChunk(
                    id=uuid4(),
                    source_id=uuid4(),
                    profile_id=uuid4(),
                    chunk_index=0,
                    text=text.rstrip("\n"),
                    line_start=1,
                    line_end=len(text.splitlines()),
                ),
                path,
            )
        )
    return SimpleNamespace(chunks=tuple(chunks), sha="b" * 40)


def request(text="Reject incomplete submissions at a web endpoint.", skill="input checks"):
    return career_retrieval.query(dict(skill=skill, aliases=[], quote=text, depth=text))


@pytest.mark.parametrize(
    "path,text,query",
    [
        ("orders.py", FLASK, "Reject incomplete submissions at a web endpoint."),
        ("edge.go", GO, "Check inbound messages before an API endpoint accepts them."),
    ],
)
def test_independent_framework_mismatch_discovers_canonical_bodies(path, text, query):
    snap = snapshot({path: text})
    req = request(query)
    from rag.retriever.revision_keyword import tokenize

    assert not set(tokenize(req.text, 16384)) & set(tokenize(text, 16384))
    seeds, hints, stats = career_structural.discover(snap, req)
    assert len(seeds) == len(hints) == 1 and stats["structural_hints"] == 1
    groups, _ = career_context.groups(snap, seeds, req, hints)
    assert groups and any("422" in w.excerpt for g in groups for w in g.windows)
    for group in groups:
        for window in group.windows:
            assert window.excerpt == "\n".join(
                text.splitlines()[window.line_start - 1 : window.line_end]
            )
            assert window.commit_sha == snap.sha
            assert len(window.excerpt.encode()) <= 300 and window.line_end - window.line_start < 8
    assert any(
        "net/http" in w.excerpt or "from flask" in w.excerpt for g in groups for w in g.windows
    )


@pytest.mark.parametrize(
    "path,text",
    [
        ("README.md", "HTTP request validation using FastAPI."),
        ("styles.css", ".request-validation { display: none; }"),
        ("notes.py", '"""from fastapi import FastAPI\napp=FastAPI()\n@app.post("/x")\n"""'),
        (
            "plain.py",
            "def validate(payload):\n    if not payload: raise ValueError()\n    return payload\n",
        ),
        (
            "fake.py",
            'class Fake: pass\napp=Fake()\n@app.post("/x")\n'
            "def accept(payload):\n    if not payload: raise ValueError()\n",
        ),
        (
            "empty.py",
            'from fastapi import FastAPI\napp=FastAPI()\n@app.post("/x")\n'
            "def accept(payload):\n    pass\n",
        ),
        ("other.go", "package edge\nfunc reject(n int) { if n < 0 { return } }\n"),
    ],
)
def test_docs_comments_stubs_and_unrelated_constructs_not_structural_seeds(path, text):
    assert not career_structural.discover(snapshot({path: text}), request())[0]


def test_absent_behavior_and_language_scope_remain_empty():
    snap = snapshot({"orders.py": FLASK, "edge.go": GO})
    assert not career_structural.discover(
        snap, request("Configure Kubernetes deployment manifests.", "kubernetes")
    )[0]
    seeds, _, _ = career_structural.discover(snap, request(skill="go"))
    assert seeds and all(c.path.endswith(".go") for c in seeds)


def test_scan_limits_are_explicit_and_seed_count_bounded(monkeypatch):
    monkeypatch.setattr(career_structural, "MAX_FILES", 1)
    snap = snapshot({"a.py": FLASK, "b.py": FLASK})
    seeds, _, stats = career_structural.discover(snap, request())
    assert (
        len(seeds) <= 2
        and stats["structural_scan_limited"]
        and stats["structural_scanned_files"] == 1
    )
    monkeypatch.setattr(career_structural, "MAX_FILE_BYTES", 10)
    assert not career_structural.discover(snap, request())[0]


def test_rebound_framework_receiver_is_not_trusted():
    text = FLASK.replace("@bp.post", "bp = object()\n@bp.post")
    assert not career_structural.discover(snapshot({"orders.py": text}), request())[0]


def test_framework_handler_without_local_callees_is_complete():
    # Deferred stabilization 2026-11-04: Flask decorator/request/abort calls are not local code.
    snap = snapshot({"orders.py": FLASK})
    req = request()
    seeds, hints, _ = career_structural.discover(snap, req)
    groups, _ = career_context.groups(snap, seeds, req, hints)
    assert groups[0].complete is True and not groups[0].missing


def test_derived_features_do_not_override_incomplete_or_checker_policy():
    # The handler now calls a locally defined helper that cannot be resolved uniquely, so the
    # group is genuinely incomplete (previously it was incomplete only because framework calls
    # were mistaken for missing local code).
    handler = FLASK.replace(
        "    return entry\n", "    entry = normalize(entry)\n    return entry\n"
    )
    snap = snapshot(
        {
            "orders.py": handler,
            "clean.py": "def normalize(entry):\n    return entry\n",
            "legacy.py": "def normalize(entry):\n    return {}\n",
        }
    )
    req = request()
    seeds, hints, _ = career_structural.discover(snap, req)
    groups, _ = career_context.groups(snap, seeds, req, hints)
    group = groups[0]
    assert group.complete is False
    assert "unresolved_or_ambiguous_call:normalize" in group.missing
    ids = ["E" + str(i + 1) for i in range(len(group.windows))]
    selected = {
        key: SimpleNamespace(**{**w.model_dump(), "required_evidence": ids})
        for key, w in zip(ids, group.windows, strict=True)
    }
    scope = {key: (str(uuid4()), str(uuid4())) for key in ids}
    raw = dict(
        classification="demonstrated",
        claim="The code checks incoming submissions.",
        references=[dict(evidence_id=ids[0])],
        uncertainty="Mocked verdict",
    )
    check = dict(classification="demonstrated", verdict="supported", complete=True)
    with pytest.raises(ValueError, match="Incomplete local implementation"):
        career_evidence.accepted(json.dumps(raw), json.dumps(check), selected, scope)


@pytest.mark.parametrize("path", ["site.css", "graphic.svg"])
def test_backend_guard_query_filters_lexically_matching_style_assets(path):
    assert career_structural.excluded_asset(path, request())
    assert not career_structural.excluded_asset(path, request("Style a page with CSS.", "css"))
    assert not career_structural.excluded_asset("README.md", request())

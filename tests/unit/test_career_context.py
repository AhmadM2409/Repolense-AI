"""Canonical local-function closure, not semantic model accuracy or source execution."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.contracts import TextChunk
from core.services import career_context as context
from core.services import career_evidence as policy
from core.services import career_retrieval


def snapshot(files, block=80):
    chunks = []
    for path, text in files.items():
        source, profile = uuid4(), uuid4()
        lines = text.splitlines()
        for i in range(0, len(lines), block):
            chunk = TextChunk(
                id=uuid4(),
                source_id=source,
                profile_id=profile,
                chunk_index=i // block,
                text="\n".join(lines[i : i + block]),
                line_start=i + 1,
                line_end=min(i + block, len(lines)),
            )
            chunks.append((chunk, path))
    return SimpleNamespace(chunks=tuple(chunks), sha="a" * 40)


def build(files, skill="python", path="reader.py", block=80):
    snap = snapshot(files, block)
    seed = [SimpleNamespace(chunk_id=c.id) for c, p in snap.chunks if p == path]
    query = career_retrieval.query(
        {"skill": skill, "aliases": [], "quote": "Read CSV records.", "depth": "Read CSV records."}
    )
    groups, flags = context.groups(snap, seed, query)
    return snap, groups, flags


def test_multifile_imports_follow_actual_definitions_not_same_name_guess():
    _, groups, _ = build(
        {
            ("reader.py"): (
                "from helpers import lookup as read_cell\n"
                "def read_csv(rows):\n"
                "    return read_cell(rows)\n"
            ),
            "helpers.py": "def lookup(rows):\n    return rows['csv']\n",
            "unrelated.py": "def lookup(rows):\n    return False\n",
        }
    )
    group = groups[0]
    assert group.complete and {w.path for w in group.windows} == {"reader.py", "helpers.py"}
    assert all(
        len(w.excerpt.encode()) <= 300 and w.line_end - w.line_start < 8 for w in group.windows
    )


@pytest.mark.parametrize("body", ["return missing(rows)", "pass", "raise NotImplementedError()"])
def test_delegation_missing_and_stub_not_complete(body):
    _, groups, _ = build({"reader.py": "def read_csv(rows):\n    " + body + "\n"})
    assert groups and not groups[0].complete and groups[0].missing


def test_opaque_external_delegation_is_not_proof():
    _, groups, _ = build(
        {"reader.py": "import library\ndef read_csv(rows):\n    return library.fetch(rows)\n"}
    )
    assert not groups[0].complete and "opaque_external_call:fetch" in groups[0].missing


def test_cycles_and_ambiguous_names_are_bounded_and_incomplete():
    _, groups, _ = build(
        {
            "reader.py": "def read_csv(rows):\n    return helper(rows)\n",
            "one.py": "def helper(rows):\n    return rows\n",
            "two.py": "def helper(rows):\n    return False\n",
        }
    )
    assert not groups[0].complete
    _, groups, _ = build(
        {
            ("reader.py"): (
                "def read_csv(rows):\n"
                "    return helper(rows)\n"
                "def helper(rows):\n"
                "    return read_csv(rows)\n"
            )
        }
    )
    assert groups and not groups[0].complete and "recursive_call" in groups[0].missing


def test_go_strings_comments_and_cross_file_call_target():
    files = {
        ("reader.go"): (
            "package reader\n"
            "func read_csv(rows map[string]interface{}) interface{} {\n"
            ' return lookup(rows, "csv")\n'
            "}\n"
        ),
        ("helper.go"): (
            "package reader\n"
            "// literal braces do not close functions: }\n"
            "func lookup(rows map[string]interface{}, key string) interfa"
            "ce{} {\n"
            ' note := "}"\n'
            " _ = note\n"
            " return rows[key]\n"
            "}\n"
        ),
    }
    _, groups, _ = build(files, "go", "reader.go")
    assert groups[0].complete and {w.path for w in groups[0].windows} == set(files)
    assert any("return rows[key]" in w.excerpt for w in groups[0].windows)


def test_large_implementation_conflict_is_explicit_not_silently_clipped():
    text = "def read_csv(rows):\n" + "    rows = rows\n" * 48 + "    return rows\n"
    _, groups, flags = build({"reader.py": text})
    assert flags["budget_conflicts"] and not groups
    assert flags["parsed_chunks"]


def test_chunk_boundary_packing_remains_exact_and_complete():
    text = "def read_csv(rows):\n    selected = rows['csv']\n    return selected\n"
    _, groups, _ = build({"reader.py": text}, block=2)
    assert groups[0].complete and len(groups[0].windows) == 2
    assert "\n".join(w.excerpt for w in groups[0].windows) == text.rstrip("\n")


def test_required_span_gate_rejects_optimistic_checker_and_allows_complete_refs():
    _, groups, _ = build(
        {
            ("reader.py"): (
                "from helpers import lookup\ndef read_csv(rows):\n    return lookup(rows)\n"
            ),
            "helpers.py": "def lookup(rows):\n    return rows['csv']\n",
        }
    )
    group = groups[0]
    ids = ["E" + str(i + 1) for i in range(len(group.windows))]
    selected = {
        key: SimpleNamespace(**{**w.model_dump(), "required_evidence": ids})
        for key, w in zip(ids, group.windows, strict=True)
    }
    identity = (str(uuid4()), str(uuid4()))
    scope = {key: identity for key in ids}
    raw = {
        "classification": "demonstrated",
        "claim": "The code defines read_csv.",
        "references": [{"evidence_id": k} for k in ids],
        "uncertainty": "Mocked semantic verdict.",
    }
    check = {"classification": "demonstrated", "verdict": "supported", "complete": True}
    assert policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)["citations"]
    raw["references"] = [{"evidence_id": ids[0]}]
    assert policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)["citations"]
    removed = selected.pop(ids[-1])
    with pytest.raises(ValueError, match="Incomplete local implementation"):
        policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)
    selected[ids[-1]] = removed
    raw["references"] = [{"evidence_id": k} for k in ids]
    selected[ids[0]].implementation_complete = False
    with pytest.raises(ValueError, match="Incomplete local implementation"):
        policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)


def test_contradictory_verdict_and_fabricated_quotes_remain_rejected():
    _, groups, _ = build({"reader.py": "def read_csv(rows):\n    return rows['csv']\n"})
    item = groups[0].windows[0]
    selected = {"E1": SimpleNamespace(**item.model_dump())}
    scopes = {"E1": (str(uuid4()), str(uuid4()))}
    raw = {
        "classification": "demonstrated",
        "claim": "The code defines read_csv.",
        "references": [{"evidence_id": "E1"}],
        "uncertainty": "Mocked checker.",
    }
    check = {"classification": "demonstrated", "verdict": "contradicted", "complete": True}
    with pytest.raises(ValueError):
        policy.accepted(json.dumps(raw), json.dumps(check), selected, scopes)
    raw["references"][0]["quote"] = "nonexistent source quotation"
    with pytest.raises(ValueError):
        policy.accepted(
            json.dumps(raw), json.dumps({**check, "verdict": "supported"}), selected, scopes
        )


def test_local_builtin_shadow_and_dynamic_parameters_are_not_assumed_native():
    _, groups, _ = build(
        {"reader.py": "def read_csv(rows):\n    return list(rows)\ndef list(rows):\n    pass\n"}
    )
    assert not groups[0].complete and "stub_implementation" in groups[0].missing
    _, groups, _ = build({"reader.py": "def read_csv(list, rows):\n    return list(rows)\n"})
    assert not groups[0].complete


def test_go_local_builtin_shadow_body_is_included():
    _, groups, _ = build(
        {
            ("reader.go"): (
                "package reader\n"
                "func read_csv(rows []string) int {\n"
                " return len(rows)\n"
                "}\n"
                "func len(rows []string) int {\n"
                " return 7\n"
                "}\n"
            )
        },
        "go",
        "reader.go",
    )
    assert groups[0].complete and len(groups[0].windows) == 2
    assert any("func len" in w.excerpt for w in groups[0].windows)


def test_go_empty_body_and_callable_parameter_are_incomplete():
    _, groups, _ = build({"reader.go": "package reader\nfunc read_csv() {}\n"}, "go", "reader.go")
    assert not groups[0].complete and "stub_implementation" in groups[0].missing
    _, groups, _ = build(
        {"reader.go": "package reader\nfunc read_csv(len func() int) int {\n return len()\n}\n"},
        "go",
        "reader.go",
    )
    assert not groups[0].complete


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "quote",
        "unknown",
        "missing",
        "foreign",
        "incomplete",
        "contradicted",
        "unrelated",
        "documentation",
        "range",
    ],
)
def test_server_group_range_resolution(fault):
    text = "def read_csv(rows):\n    value = rows['csv']\n    return value\n"
    _, groups, _ = build({"reader.py": text}, block=2)
    group = groups[0]
    ids = ["E" + str(i + 1) for i in range(len(group.windows))]
    selected = {
        key: SimpleNamespace(**{**w.model_dump(), "required_evidence": ids})
        for key, w in zip(ids, group.windows, strict=True)
    }
    for item in selected.values():
        item.chunk_id = selected[ids[0]].chunk_id
    identity = (str(uuid4()), str(uuid4()))
    scope = {key: identity for key in ids}
    raw = {
        "classification": "demonstrated",
        "claim": "The code defines read_csv.",
        "references": [{"evidence_id": ids[0], "quote": "def read_csv(rows):"}],
        "uncertainty": "Mocked semantic checker, not live inference.",
    }
    check = {"classification": "demonstrated", "verdict": "supported", "complete": True}
    if fault == "quote":
        raw["references"][0]["quote"] = "invented text"
    if fault == "unknown":
        raw["references"][0]["evidence_id"] = "E99"
    if fault == "missing":
        selected.pop(ids[-1])
    if fault == "foreign":
        scope[ids[-1]] = (str(uuid4()), str(uuid4()))
    if fault == "incomplete":
        selected[ids[-1]].implementation_complete = False
    if fault == "contradicted":
        check["verdict"] = "contradicted"
    if fault == "unrelated":
        raw["claim"] = "The code implements Kubernetes orchestration."
        check["verdict"] = "insufficient"
    if fault == "documentation":
        for item in selected.values():
            item.path = "README.md"
    if fault == "range":
        selected[ids[-1]].line_end += 1
    if fault is None:
        # A retrieved but unselected unrelated group must never be attached.
        selected["E99"] = SimpleNamespace(**vars(selected[ids[0]]))
        selected["E99"].required_evidence = ["E99"]
        selected["E99"].path = "unrelated.py"
        selected["E99"].source_id = uuid4()
        scope["E99"] = identity
    if fault:
        with pytest.raises(ValueError):
            policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)
    else:
        result = policy.accepted(json.dumps(raw), json.dumps(check), selected, scope)
        assert len(result["citations"]) == 1
        citation = result["citations"][0]
        assert (citation["line_start"], citation["line_end"]) == (1, 3)
        assert citation["quote"] == text.rstrip("\n")


# Deferred-stabilization regression (2026-11-04): calls whose targets are not defined anywhere in
# the verified revision (builtins, stdlib/third-party objects, framework decorators) are not
# missing local implementation. Locally defined but unresolvable targets stay incomplete.
_HANDLER = (
    "from fastapi import FastAPI, HTTPException\n"
    "app = FastAPI()\n"
    "@app.post('/orders')\n"
    "def read_csv(payload):\n"
    "    if not payload.get('items'):\n"
    "        raise {error}('empty order')\n"
    "    return {{'accepted': True}}\n"
)


@pytest.mark.parametrize("error", ["ValueError", "HTTPException"])
def test_framework_decorator_builtin_and_third_party_calls_are_not_missing_local_code(error):
    _, groups, _ = build({"handler.py": _HANDLER.format(error=error)}, path="handler.py")
    assert groups and groups[0].complete and not groups[0].missing


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {
                "handler.py": "def read_csv(store, item):\n    store.save(item)\n    return True\n",
                "models.py": "class Store:\n    def save(self, item):\n        return item\n",
            },
            "unresolved_or_ambiguous_call:save",
        ),
        (
            {
                "handler.py": (
                    "from .rules import check\n"
                    "def read_csv(item):\n    check(item)\n    return True\n"
                )
            },
            "unresolved_or_ambiguous_call:check",
        ),
        (
            {"handler.py": "def read_csv(item):\n    value = helper(item)\n    return value\n"},
            "unresolved_or_ambiguous_call:helper",
        ),
        (
            {"handler.py": "def read_csv(item):\n    return item.process()\n"},
            "opaque_external_call:process",
        ),
    ],
)
def test_local_or_unknown_targets_and_opaque_delegation_remain_incomplete(files, expected):
    _, groups, _ = build(files, path="handler.py")
    assert groups and not groups[0].complete and expected in groups[0].missing


def test_go_method_on_nonlocal_value_is_not_missing_but_local_method_is_followed():
    files = {
        "reader.go": (
            "package reader\n"
            'import "strings"\n'
            "func read_csv(rows []string) string {\n"
            " var b strings.Builder\n"
            " b.WriteString(rows[0])\n"
            " return format(b.String())\n"
            "}\n"
            "func format(s string) string {\n"
            ' return s + "!"\n'
            "}\n"
        )
    }
    _, groups, _ = build(files, "go", "reader.go")
    assert groups[0].complete and any("func format" in w.excerpt for w in groups[0].windows)


_READER = "from limits import MAX_ROWS\ndef read_csv(rows):\n    return len(rows) < MAX_ROWS\n"


def test_imported_local_constant_definition_is_included_as_evidence():
    # Deferred stabilization 2026-11-04: a compared/returned constant imported from a local
    # module is shown with its definition instead of leaving its value out of the evidence.
    _, groups, _ = build({"reader.py": _READER, "limits.py": "OTHER = 1\nMAX_ROWS = 67\n"})
    group = groups[0]
    assert group.complete
    excerpts = {w.path: w.excerpt for w in group.windows}
    assert "from limits import MAX_ROWS" in excerpts["reader.py"]
    assert excerpts["limits.py"] == "MAX_ROWS = 67"


@pytest.mark.parametrize(
    "limits",
    ["def MAX_ROWS():\n    return 67\n", "MAX_ROWS: int\n", None],
)
def test_unresolvable_imported_value_adds_nothing_and_changes_nothing(limits):
    files = {"reader.py": _READER}
    if limits is not None:
        files["limits.py"] = limits
    _, groups, _ = build(files)
    assert {w.path for w in groups[0].windows} == {"reader.py"}

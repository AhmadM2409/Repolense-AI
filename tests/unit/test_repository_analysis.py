# Token-shaped values are synthetic security fixtures, never real credentials.
import hashlib
from uuid import uuid4

import pytest

from analysis.engine import analyze
from analysis.rules import RULES, python_rules
from core.contracts import SourceDocument
from tests.analysis_fixtures import FILES

CASES = [
    (
        "PY-BARE-EXCEPT",
        "try:\n    pass\nexcept:\n    pass\n",
        "try:\n    pass\nexcept Exception:\n    pass\n",
        "app.py",
    ),
    (
        "PY-SHELL-TRUE",
        "import subprocess as sp\nsp.run(value,shell=True)\n",
        "import subprocess as sp\nsp.run(['echo','safe'],shell=False)\n",
        "app.py",
    ),
    (
        "PY-SQL-DYNAMIC",
        "cursor.execute(f'SELECT * FROM users WHERE id={value}')\n",
        "cursor.execute('SELECT * FROM users WHERE id=?',(value,))\n",
        "app.py",
    ),
    (
        "PY-DESERIALIZE",
        "import pickle as p\np.loads(data)\n",
        "import json\njson.loads(data)\n",
        "app.py",
    ),
    ("PY-DEV-CONFIG", "DEBUG = True\n", "DEBUG = False\n", "settings.py"),
    (
        "PY-SECRET-LITERAL",
        "API_TOKEN = 'ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4'\n",
        "API_TOKEN = 'your_example_token_placeholder'\n",
        "app.py",
    ),
]


@pytest.mark.parametrize("rule,unsafe,safe,path", CASES)
def test_each_rule_and_safeguard(rule, unsafe, safe, path):
    _, hits = python_rules(path, unsafe)
    assert rule in {hit.rule_id for hit in hits}
    assert rule not in {hit.rule_id for hit in python_rules(path, safe)[1]}
    assert RULES[rule][3] in {"potential", "observation"}


@pytest.mark.parametrize(
    "text",
    [
        "import yaml\nyaml.load(data,Loader=yaml.SafeLoader)\n",
        "from yaml import load, SafeLoader\nload(data,Loader=SafeLoader)\n",
        "import subprocess\nsubprocess=other\nsubprocess.run(value,shell=True)\n",
        "import subprocess\ndef f(subprocess):\n    subprocess.run(value,shell=True)\n",
        "# subprocess.run(value,shell=True)\ntext='shell=True'\n",
        "import uvicorn\nif __name__ == '__main__':\n    uvicorn.run('app:app',reload=True)\n",
    ],
)
def test_false_positive_suppression(text):
    assert python_rules("app.py", text)[1] == []


@pytest.mark.parametrize(
    "text",
    [
        "from pickle import loads as parse\nparse(data)\n",
        "import yaml\nyaml.load(data)\n",
        "import marshal\nmarshal.loads(data)\n",
    ],
)
def test_deserializer_aliases(text):
    assert any(hit.rule_id == "PY-DESERIALIZE" for hit in python_rules("app.py", text)[1])


def test_secret_suppressed_in_test_fixtures_and_environment_lookup():
    text = "API_KEY = 'ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4'\n"
    assert not python_rules("tests/fixture.py", text)[1]
    assert not python_rules("app.py", "import os\nAPI_KEY=os.environ['API_KEY']\n")[1]


def docs(files):
    pid, rev = uuid4(), uuid4()
    return pid, [
        SourceDocument(
            id=uuid4(),
            profile_id=pid,
            revision_id=rev,
            source_type="repo",
            path=path,
            text=raw.decode(),
            content_hash=hashlib.sha256(raw).hexdigest(),
        )
        for path, raw in files.items()
    ]


def test_multifile_architecture_dependencies_and_exact_evidence():
    pid, documents = docs(FILES)
    repo, job = uuid4(), uuid4()
    summary, findings = analyze(documents, repo, "a" * 40, job, pid)
    assert summary["languages"]["Python"] == 2
    assert {item["name"] for item in summary["dependencies"]} >= {
        "fastapi",
        "pytest",
        "react",
        "vitest",
    }
    assert {item["name"] for item in summary["framework_indicators"]} >= {
        "FastAPI",
        "React",
        "pytest",
    }
    assert summary["test_locations"] == ["tests/test_app.py"]
    assert summary["ci_configurations"][0]["path"] == ".github/workflows/ci.yml"
    assert summary["deployment_configurations"][0]["path"] == "Dockerfile"
    assert summary["entrypoint_candidates"]
    assert {finding.rule_id for finding in findings} == {
        "PY-SHELL-TRUE",
        "PY-SQL-DYNAMIC",
        "PY-DESERIALIZE",
        "PY-BARE-EXCEPT",
    }
    by_id = {doc.id: doc for doc in documents}
    for finding in findings:
        for span in finding.evidence:
            source = by_id[span.source_id]
            assert span.quote == "\n".join(
                source.text.splitlines()[span.line_start - 1 : span.line_end]
            )
            assert span.path == source.path and span.commit_sha == "a" * 40
    again = analyze(documents, repo, "a" * 40, job, pid)
    assert [f.model_dump(mode="json") for f in again[1]] == [
        f.model_dump(mode="json") for f in findings
    ]


def test_zero_findings_is_not_a_safety_score():
    pid, documents = docs({"app.py": b"def add(a,b):\n    return a+b\n"})
    summary, findings = analyze(documents, uuid4(), "a" * 40, uuid4(), pid)
    assert findings == [] and "score" not in summary


def test_invalid_python_and_manifests_are_coverage_diagnostics():
    pid, documents = docs({"app.py": b"def broken(:\n", "package.json": b'{"dependencies":42}'})
    summary, findings = analyze(documents, uuid4(), "a" * 40, uuid4(), pid)
    assert {item["code"] for item in summary["diagnostics"]} == {
        "python_rules_not_run",
        "manifest_not_fully_parsed",
    }
    assert not findings


def test_multiline_shell_evidence_contains_keyword_not_just_start_of_call():
    text = "import subprocess\nsubprocess.run(\n    value,\n" + "\n" * 10 + "    shell=True,\n)\n"
    pid, documents = docs({"app.py": text.encode()})
    _, findings = analyze(documents, uuid4(), "a" * 40, uuid4(), pid)
    assert "shell=True" in findings[0].evidence[0].quote


def test_manifest_comment_is_not_used_as_entrypoint_evidence():
    pid, documents = docs(
        {
            "pyproject.toml": b'# sample comment\n[project]\nname="fixture"\ndependencies=[]\n[project.scripts]\nsample="pkg:main"\n'  # noqa: E501 - exact source fixture
        }
    )
    summary, _ = analyze(documents, uuid4(), "a" * 40, uuid4(), pid)
    candidate = summary["entrypoint_candidates"][0]
    assert candidate["target"] == "pkg:main"
    assert candidate["evidence"][0]["line_start"] == 6
    assert candidate["evidence"][0]["quote"] == 'sample="pkg:main"'


def test_relative_module_import_is_not_assumed_to_be_pickle():
    assert python_rules("app.py", "from .pickle import loads\nloads(data)\n")[1] == []


def test_invalid_dependency_list_does_not_invent_packages():
    pid, documents = docs({"pyproject.toml": b'[project]\ndependencies="fastapi"\n'})
    summary, _ = analyze(documents, uuid4(), "a" * 40, uuid4(), pid)
    assert summary["dependencies"] == []
    assert summary["diagnostics"]


def test_rebound_module_member_and_low_entropy_placeholder_are_suppressed():
    assert (
        python_rules(
            "app.py", "import subprocess\nsubprocess.run = safe\nsubprocess.run(value,shell=True)\n"
        )[1]
        == []
    )
    assert python_rules("app.py", "API_TOKEN='ghp_AAAAAAAAAAAAAAAAAAAAAAAAAA'\n")[1] == []


def test_conflicting_import_aliases_are_not_assumed_to_be_unsafe_modules():
    text = "import json as parser\nparser.loads(data)\nimport pickle as parser\n"
    assert python_rules("app.py", text)[1] == []

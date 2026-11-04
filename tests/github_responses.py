"""Deterministic real-shape GitHub Git-object HTTP fixtures (external dependency only)."""

import base64
import hashlib
import httpx


def object_hash(kind, data):
    return hashlib.sha1((kind + " " + str(len(data))).encode() + b"\0" + data).hexdigest()


class GitHubResponses:
    def __init__(self, files=None, *, branch="main", name="example/project"):
        self.name = name
        self.branch = branch
        self.requests = []
        self.overrides = {}
        self.blobs = {}
        self.trees = {}
        root = {}
        for path, spec in (
            files
            or {
                "README.md": b"# Real fixture\r\nContent\r\n",
                "src/main.py": b"print('fixture')\n",
                "package.json": b'{"scripts":{"test":"pytest"}}',
            }
        ).items():
            raw, mode = spec if isinstance(spec, tuple) else (spec, "100644")
            parts = path.split("/")
            node = root
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = (raw, mode)

        def tree(node):
            entries = []
            for path, value in node.items():
                if isinstance(value, dict):
                    oid = tree(value)
                    entries.append({"path": path, "mode": "040000", "type": "tree", "sha": oid})
                else:
                    raw, mode = value
                    oid = object_hash("blob", raw)
                    self.blobs[oid] = {
                        "sha": oid,
                        "size": len(raw),
                        "encoding": "base64",
                        "content": base64.b64encode(raw).decode() + "\n",
                    }
                    entries.append(
                        {"path": path, "mode": mode, "type": "blob", "sha": oid, "size": len(raw)}
                    )
            ordered = sorted(
                entries, key=lambda x: x["path"].encode() + (b"/" if x["type"] == "tree" else b"")
            )
            raw = b"".join(
                e["mode"].lstrip("0").encode()
                + b" "
                + e["path"].encode()
                + b"\0"
                + bytes.fromhex(e["sha"])
                for e in ordered
            )
            oid = object_hash("tree", raw)
            self.trees[oid] = {"sha": oid, "tree": entries, "truncated": False}
            return oid

        self.tree_sha = tree(root)
        self.sha = object_hash(
            "commit", ("tree " + self.tree_sha + "\n\nSynthetic test commit\n").encode()
        )

    def __call__(self, request):
        self.requests.append(request)
        assert request.url.host == "api.github.com" and request.url.scheme == "https"
        path = request.url.path
        suffix = "" if path == "/repos/" + self.name else path.removeprefix("/repos/" + self.name + "/")
        if suffix in self.overrides:
            override = self.overrides[suffix]
            return override(request) if callable(override) else httpx.Response(200, json=override)
        if suffix == "":
            data = {
                "full_name": self.name,
                "private": False,
                "visibility": "public",
                "default_branch": self.branch,
                "url": "http://127.0.0.1/never-follow",
            }
        elif suffix == "git/ref/heads/" + self.branch:
            data = {
                "ref": "refs/heads/" + self.branch,
                "object": {"type": "commit", "sha": self.sha},
            }
        elif suffix == "git/commits/" + self.sha:
            data = {"sha": self.sha, "tree": {"sha": self.tree_sha}}
        elif suffix.startswith("git/trees/") and suffix[10:] in self.trees:
            data = self.trees[suffix[10:]]
        elif suffix.startswith("git/blobs/") and suffix[10:] in self.blobs:
            data = self.blobs[suffix[10:]]
        else:
            return httpx.Response(404, json={"message": "Not found"})
        return httpx.Response(200, json=data)

"""Server-only Gemini configuration and persistent fail-closed task/runtime accounting."""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from core.config import settings
from rag.generator.gemini_provider import MODEL, URL, Gemini, load_secret
from rag.generator.local_provider import GenerationError

CONSENT = "gemini-source-processing-v1"
ROOT = Path(__file__).resolve().parents[2]


def credential():
    key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", MODEL)
    if key:
        if model != MODEL:
            raise GenerationError("configuration")
        return key
    try:
        return load_secret(ROOT)
    except OSError:
        raise GenerationError("configuration") from None


def identity():
    if not settings.qa_gemini_enabled:
        raise GenerationError("configuration")
    return hashlib.sha256(
        json.dumps(
            {
                "model": MODEL,
                "endpoint": URL,
                "key_digest": hashlib.sha256(credential().encode()).hexdigest(),
                "ledger": settings.qa_gemini_ledger_path,
                "stage": settings.qa_gemini_budget_stage,
                "enabled": settings.qa_gemini_enabled,
                "policy": "grounded-claims-v4",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


class PersistentBudget:
    """Atomic process-safe counter; no prompts, answer text or receipts written to disk.

    A ledger is not reset on process restart. Limits are fixed: 100 total, 60 development,
    40 acceptance, 10 attempts per rolling minute. Errors consume reservations.
    """

    def __init__(self, path, stage="development"):
        if stage not in {"development", "acceptance"}:
            raise GenerationError("configuration")
        self.path = Path(path)
        self.stage = stage
        self.entries = []
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, "
                    "at REAL NOT NULL, stage TEXT NOT NULL, metadata TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS control (id INTEGER PRIMARY KEY "
                    "CHECK(id=1), stopped INTEGER NOT NULL)"
                )
                db.execute("INSERT OR IGNORE INTO control VALUES(1,0)")
        except (OSError, sqlite3.Error):
            raise GenerationError("budget_storage") from None

    def connect(self):
        return sqlite3.connect(self.path, timeout=2)

    @property
    def stopped(self):
        with self.connect() as db:
            return bool(db.execute("SELECT stopped FROM control WHERE id=1").fetchone()[0])

    @stopped.setter
    def stopped(self, value):
        if value:
            with self.connect() as db:
                db.execute("UPDATE control SET stopped=1 WHERE id=1")

    def reserve(self):
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT stopped FROM control WHERE id=1").fetchone()[0]:
                    raise GenerationError("quota_or_billing")
                count = db.execute("SELECT count(*) FROM requests").fetchone()[0]
                stage_count = db.execute(
                    "SELECT count(*) FROM requests WHERE stage=?", (self.stage,)
                ).fetchone()[0]
                if count >= 100 or stage_count >= (60 if self.stage == "development" else 40):
                    raise GenerationError("request_budget")
                now = time.time()
                if (
                    db.execute("SELECT count(*) FROM requests WHERE at>?", (now - 60,)).fetchone()[
                        0
                    ]
                    >= 10
                ):
                    raise GenerationError("capacity")
                cursor = db.execute(
                    "INSERT INTO requests(at,stage,metadata) VALUES(?,?,?)",
                    (now, self.stage, '{"state":"reserved_before_post"}'),
                )
                row = {"number": cursor.lastrowid, "state": "reserved_before_post"}
            self.entries.append(row)
            return row
        except sqlite3.Error:
            raise GenerationError("budget_storage") from None

    def save(self):
        # The shared adapter keeps bounded response diagnostics in memory. Production
        # persists only operational metadata, never content or receipt/source text.
        names = {
            "number",
            "state",
            "http_status",
            "model",
            "finish_reason",
            "usage",
            "input_bytes",
            "latency_seconds",
            "validation_stage",
            "rejection_reason",
        }
        try:
            with self.connect() as db:
                for row in self.entries:
                    db.execute(
                        "UPDATE requests SET metadata=? WHERE id=?",
                        (json.dumps({k: v for k, v in row.items() if k in names}), row["number"]),
                    )
        except sqlite3.Error:
            raise GenerationError("budget_storage") from None


def make_provider():
    identity()
    return Gemini(
        credential(),
        PersistentBudget(settings.qa_gemini_ledger_path, settings.qa_gemini_budget_stage),
    )

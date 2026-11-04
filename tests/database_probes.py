"""Real database failure probes with equivalent SQLite/PostgreSQL semantics."""

import re
from sqlalchemy import text


async def reject_insert(db, name: str, table: str, message: str):
    if not all(re.fullmatch(r"[a-z_]+", value) for value in (name, table)):
        raise ValueError("Probe identifiers must be static SQL identifiers")
    message = message.replace("'", "''")
    if db.bind.dialect.name == "postgresql":
        await db.execute(
            text(
                "CREATE FUNCTION " + name + "_probe() RETURNS trigger LANGUAGE plpgsql AS $$ "
                "BEGIN RAISE EXCEPTION '" + message + "' USING ERRCODE = '23514'; END; $$"
            )
        )
        await db.execute(
            text(
                "CREATE TRIGGER "
                + name
                + " BEFORE INSERT ON "
                + table
                + " FOR EACH ROW EXECUTE FUNCTION "
                + name
                + "_probe()"
            )
        )
    else:
        await db.execute(
            text(
                "CREATE TRIGGER "
                + name
                + " BEFORE INSERT ON "
                + table
                + " BEGIN SELECT RAISE(ABORT, '"
                + message
                + "'); END"
            )
        )


async def revoke_processing_job(db, action: str):
    """Action contains only generated fixture UUIDs and static test SQL."""
    if db.bind.dialect.name == "postgresql":
        await db.execute(
            text(
                "CREATE FUNCTION revoke_job_probe() RETURNS trigger LANGUAGE plpgsql AS $$ "
                "BEGIN IF NEW.status='processing' THEN " + action + " END IF; RETURN NEW; END; $$"
            )
        )
        await db.execute(
            text(
                "CREATE TRIGGER revoke_job AFTER UPDATE OF status ON reviews "
                "FOR EACH ROW EXECUTE FUNCTION revoke_job_probe()"
            )
        )
    else:
        await db.execute(
            text(
                "CREATE TRIGGER revoke_job AFTER UPDATE OF status ON reviews "
                "WHEN NEW.status='processing' BEGIN " + action + " END"
            )
        )

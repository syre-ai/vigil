"""SQLite-backed mutable state: the `reviews` table.

Per API.md: the only mutable state the backend owns is
`reviews(source TEXT DEFAULT 'gt', ann_id, action, label, note, ts, version_applied, predicted_role,
predicted_version, PRIMARY KEY(source, ann_id))`. Version history itself lives in
data/scores/versions.json, written by vigil.pipeline.build_version -- this module never writes that
file.

`source` (field mode, research/08-field-mode-proposal.md) distinguishes a review against a
ground-truth chip (`source='gt'`, the historical/benchmark case -- `ann_id` is a real
annotations.parquet ann_id) from a review against a detector chip (`source='det'` -- `ann_id` here
actually carries a det_id; see backend/API.md's "keep the API's ann_id naming" convention). The two
id spaces are NOT shared (a gt ann_id and a det det_id can collide numerically), which is why the
primary key is the PAIR (source, ann_id), not ann_id alone.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    source TEXT NOT NULL DEFAULT 'gt',
    ann_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    label TEXT,
    note TEXT,
    ts TEXT NOT NULL,
    version_applied TEXT,
    predicted_role TEXT,
    predicted_version TEXT,
    PRIMARY KEY (source, ann_id)
)
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _migrate_add_source(conn: sqlite3.Connection) -> None:
    """Migrates a reviews table created before the `source` column existed (single-source schema,
    PRIMARY KEY(ann_id) alone). SQLite can't ALTER a primary key in place, so this rebuilds the
    table: every pre-migration row is a GT-chip review (field mode's det-chip reviews didn't exist
    yet at that schema version), so source='gt' for all of them. Existing dev/prod databases (e.g.
    a running server's data/vigil.db) must keep working without a manual reset.
    """
    conn.execute("ALTER TABLE reviews RENAME TO reviews_pre_source")
    conn.execute(_SCHEMA)
    conn.execute(
        """
        INSERT INTO reviews (source, ann_id, action, label, note, ts, version_applied, predicted_role, predicted_version)
        SELECT 'gt', ann_id, action, label, note, ts, version_applied, predicted_role, predicted_version
        FROM reviews_pre_source
        """
    )
    conn.execute("DROP TABLE reviews_pre_source")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Migrates a reviews table created before predicted_role/predicted_version/source existed.
    SQLite has no "ADD COLUMN IF NOT EXISTS" (prior to very recent versions), so check PRAGMA
    table_info first. Order matters: predicted_role/predicted_version are added (if missing) to the
    table under its CURRENT name before _migrate_add_source potentially renames it away, so the
    source-migration's INSERT...SELECT can always reference those columns.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(reviews)")}
    if not cols:
        return  # table was just created fresh by _SCHEMA (CREATE TABLE IF NOT EXISTS), above
    if "predicted_role" not in cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN predicted_role TEXT")
    if "predicted_version" not in cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN predicted_version TEXT")
    if "source" not in cols:
        _migrate_add_source(conn)


def get_conn() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    _ensure_columns(conn)
    conn.commit()
    return conn


def review_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "source": row["source"],
        "ann_id": row["ann_id"],
        "action": row["action"],
        "label": row["label"],
        "note": row["note"],
        "ts": row["ts"],
        "version_applied": row["version_applied"],
        "predicted_role": row["predicted_role"],
        "predicted_version": row["predicted_version"],
    }


def upsert_review(
    ann_id: int,
    action: str,
    label: str | None,
    note: str | None,
    predicted_role: str | None = None,
    predicted_version: str | None = None,
    source: str = "gt",
) -> dict[str, Any]:
    """Records (or overwrites, if this (source, ann_id) was already reviewed) one review.

    `predicted_role`/`predicted_version` snapshot the chip's CURRENT role_pred (from the GT scores
    parquet when source='gt', from the det scores parquet when source='det' -- see
    backend/app/main.py's post_review) and the CURRENT version id at the moment of this POST --
    overwritten on every call, same as action/label/note/ts, so a re-review always reflects what the
    analyst was actually looking at. Used as the "from" side of analyst-disagreement evidence (see
    dataio.analyst_disagreement) -- never used to alter the model.
    """
    conn = get_conn()
    try:
        ts = now_iso()
        conn.execute(
            """
            INSERT INTO reviews (source, ann_id, action, label, note, ts, version_applied, predicted_role, predicted_version)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
            ON CONFLICT(source, ann_id) DO UPDATE SET
                action=excluded.action,
                label=excluded.label,
                note=excluded.note,
                ts=excluded.ts,
                version_applied=NULL,
                predicted_role=excluded.predicted_role,
                predicted_version=excluded.predicted_version
            """,
            (source, ann_id, action, label, note, ts, predicted_role, predicted_version),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM reviews WHERE source=? AND ann_id=?", (source, ann_id)).fetchone()
        return review_to_dict(row)
    finally:
        conn.close()


def all_reviews(source: str | None = None) -> list[dict[str, Any]]:
    """All reviews, most-recently-reviewed first. `source` restricts to 'gt' or 'det'; None (or
    'all') pools BOTH sources -- the historical, pre-field-mode behavior, and the default for
    GET /review/log and for dataio.analyst_disagreement (which cares about confirm/correct actions
    regardless of which kind of chip they were recorded against)."""
    conn = get_conn()
    try:
        if source and source != "all":
            rows = conn.execute("SELECT * FROM reviews WHERE source=? ORDER BY ts DESC", (source,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM reviews ORDER BY ts DESC").fetchall()
        return [review_to_dict(r) for r in rows]
    finally:
        conn.close()


def reviews_by_ann_id(source: str = "gt") -> dict[int, dict[str, Any]]:
    """ann_id -> review dict, restricted to ONE source (default 'gt', the historical/benchmark
    behavior every existing call site relies on). ann_id/det_id numbering is NOT shared between the
    two source namespaces, so pooling both sources here would silently collide a gt chip and a det
    chip that happen to share a numeric id -- callers that need det-chip reviews pass source='det'
    explicitly (see backend/app/main.py's get_det_chip)."""
    return {r["ann_id"]: r for r in all_reviews(source=source)}


def distinct_labels() -> list[str]:
    """Distinct non-null labels ever used in a review, most-recently-used first -- pools BOTH
    sources by construction (no source filter): a label is a label regardless of which kind of chip
    it was assigned to (see API.md's GET /review/labels, "both sources")."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT label, MAX(ts) AS last_used
            FROM reviews
            WHERE label IS NOT NULL
            GROUP BY label
            ORDER BY last_used DESC
            """
        ).fetchall()
        return [r["label"] for r in rows]
    finally:
        conn.close()


def unapplied_reviews() -> list[dict[str, Any]]:
    """Every review (either source) not yet folded into a built version -- /adapt's feedback list.
    Each item carries `source` (via review_to_dict), so vigil.pipeline._apply_feedback can route it
    to the right embedding space (GT vs detector chip) and mark_applied can scope its UPDATE by
    (source, ann_id)."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE version_applied IS NULL ORDER BY ts ASC"
        ).fetchall()
        return [review_to_dict(r) for r in rows]
    finally:
        conn.close()


def mark_applied(items: list[tuple[str, int]], version_id: str) -> None:
    """Marks each (source, ann_id) pair's review as applied to version_id. Scoped by BOTH source and
    ann_id (not ann_id alone) so a gt review and a det review that happen to share a numeric id are
    never cross-updated."""
    if not items:
        return
    conn = get_conn()
    try:
        conn.executemany(
            "UPDATE reviews SET version_applied=? WHERE source=? AND ann_id=?",
            [(version_id, source, ann_id) for source, ann_id in items],
        )
        conn.commit()
    finally:
        conn.close()

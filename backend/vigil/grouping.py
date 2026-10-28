"""Field-mode collection grouping key, shared by the API layer (backend/app/collections.py) and the
pipeline (backend/vigil/field.py) so a `group_id` computed offline (eval_{version}.json's
field.classes[].by_group) matches the one `/api/collections` serves at request time.

Deliberately dependency-free (stdlib only), like vigil.paths -- importing this never pulls in pandas.

A "collection" is one airfield x collection date (UTC day) group of new, unlabeled parent images --
never a single parent image, never a ground-truth batch/domain name (STATE.md's Field mode consensus).
"""

from __future__ import annotations

import re

GROUPING = "airfield × collection date (UTC day)"


def slug(value: object) -> str:
    """Lowercases and collapses every run of non-alphanumerics to a single '-'
    (e.g. "Some Airfield, TX" -> "some-airfield-tx"); "unknown" for empty/None."""
    s = str(value or "unknown").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "unknown"


def group_id(airfield: object, date: str) -> str:
    """`slug(airfield) + "_" + date` -- url-safe, stable. `date` is YYYY-MM-DD (UTC day)."""
    return f"{slug(airfield)}_{date}"

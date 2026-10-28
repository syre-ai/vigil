"""Shared NaN/Infinity -> None sanitizer for JSON serialization.

Plain `json.dumps` happily emits Python's `NaN`/`Infinity` literals for
float('nan')/float('inf'), which round-trip through Python's own `json`
module but are not valid JSON and are rejected by most frontend
`JSON.parse` implementations. `json_safe()` recursively replaces both with
`None` so every JSON file this codebase writes is spec-compliant.

Deliberately dependency-free (stdlib only) so importing this module alone
never pulls in numpy/pandas/torch/etc -- vigil.pipeline, vigil.evaluate, and
backend.app.dataio (`_sanitize_nan`) all import `json_safe` from here
instead of keeping their own copies.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively replaces float NaN/Infinity with None so json.dumps produces standard, portable JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

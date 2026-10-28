"""Exercises the adaptation pipeline end to end on a SCRATCH version, "rehearsal_v2", without ever
persisting it where the live app (or the live app's servers, currently in use) can see it.

WHY the extra care around versions.json: `vigil.pipeline.build_version()` is the only public entry
point that builds a version, and it has a side effect -- it appends (or replaces) an entry in
data/scores/versions.json, which `current_version()` (backend/app/dataio.py) reads as "the last id in
that list". If we let it run unguarded, a live request during our build (there is a real one, right
now) would briefly see `current_version() == "rehearsal_v2"` and 404/500 on `eval_rehearsal_v2.json`
before evaluate() has even finished writing it. So `_save_version_entry` is monkeypatched to a no-op
for the duration of the build -- rehearsal_v2 never becomes visible as `current_version` to the live
app, not even transiently -- and every rehearsal_v2-specific file this script's own build_version()
call wrote under data/scores/ is deleted again at the end (see `_cleanup`), copying the small,
useful ones to artifacts/rehearsal_v2/ (git-tracked) first. Net effect: this script's only lasting
changes are inside artifacts/rehearsal_v2/; data/scores/ and versions.json end exactly as they started
(v1 only).

Feedback exercised (both `new_class` and `confirm`, matching backend/API.md's `/review` actions):
the 6 highest-novelty true military-fighter chips (ground truth role_id 6, "Military
Fighter/Interceptor/Attack" -- see vigil.evaluate.MILITARY_ROLE_NAMES) in the "military" batch of v1,
labeled new_class "Military fighter" (an analyst spotting the most novel-looking real fighters first,
the way the review queue's novelty-sorted strategy would surface them); the NEXT 2 highest-novelty
true-fighter chips, as confirm (recorded agreement only, no model effect -- CONFIRM_POLICY_NOTE).

Run: .venv/bin/python scripts/exercise_adaptation.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402

from vigil import pipeline  # noqa: E402
from vigil.data import load_annotations  # noqa: E402
from vigil.evaluate import MILITARY_ROLE_NAMES  # noqa: E402

SCORES_DIR = pipeline.SCORES_DIR
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "rehearsal_v2"
VERSION_ID = "rehearsal_v2"
BASE_VERSION_ID = "v1"
NEW_LABEL = "Military fighter"
FIGHTER_ROLE_ID = 6  # MILITARY_ROLE_NAMES[6] == "Military Fighter/Interceptor/Attack"
N_NEW_CLASS = 6
N_CONFIRM = 2

assert MILITARY_ROLE_NAMES[FIGHTER_ROLE_ID] == "Military Fighter/Interceptor/Attack"


def _pick_feedback() -> list[dict]:
    """Picks the N_NEW_CLASS highest-novelty true-fighter chips in v1's "military" batch as new_class
    feedback (label NEW_LABEL), and the next N_CONFIRM highest-novelty true-fighter chips as confirm
    feedback. Ground truth (role_id) is used only to build a reproducible, demonstrative feedback set
    -- exactly the chips an analyst sorting the review queue by novelty would see first among the real
    fighters, not to cheat the model itself (the model only ever sees ann_id + label, same as a real
    /review POST)."""
    v1 = pd.read_parquet(SCORES_DIR / f"{BASE_VERSION_ID}.parquet")
    anns = load_annotations()[["ann_id", "role_id"]]
    df = v1[v1["batch_id"] == "military"].merge(anns, on="ann_id", how="left")
    fighters = df[df["role_id"] == FIGHTER_ROLE_ID].sort_values("novelty", ascending=False)
    need = N_NEW_CLASS + N_CONFIRM
    if len(fighters) < need:
        raise SystemExit(f"need >= {need} true-fighter chips in v1's military batch, found {len(fighters)}")
    top = fighters.iloc[:N_NEW_CLASS]
    next_ = fighters.iloc[N_NEW_CLASS:need]
    feedback = [{"action": "new_class", "label": NEW_LABEL, "ann_id": int(a)} for a in top["ann_id"]]
    feedback += [{"action": "confirm", "ann_id": int(a)} for a in next_["ann_id"]]
    return feedback


def _build_scratch_version(feedback: list[dict]) -> dict:
    """Calls the real vigil.pipeline.build_version(), with versions.json writes suppressed for the
    duration (see module docstring) -- restored in `finally` regardless of outcome."""
    original_save = pipeline._save_version_entry
    pipeline._save_version_entry = lambda entry: None
    try:
        return pipeline.build_version(VERSION_ID, feedback=feedback, note="rehearsal exercise (scratch, deleted after run)")
    finally:
        pipeline._save_version_entry = original_save


def _print_compare(eval_a: dict, eval_b: dict) -> None:
    """Prints /versions/compare-style numbers computed from the two eval_{version}.json dicts."""
    print(f"\n=== {BASE_VERSION_ID} vs {VERSION_ID} (compare-style) ===")

    pc_a = {name: v.get("accuracy") for name, v in eval_a["role_head"]["per_class"].items()}
    pc_b = {name: v.get("accuracy") for name, v in eval_b["role_head"]["per_class"].items()}
    print("per-class control (ref_holdout) accuracy:")
    for name in sorted(set(pc_a) | set(pc_b)):
        print(f"  {name:14s} {BASE_VERSION_ID}={pc_a.get(name)}  {VERSION_ID}={pc_b.get(name)}")
    print(f"  overall         {BASE_VERSION_ID}={eval_a['role_head']['accuracy_ref_holdout']}  "
          f"{VERSION_ID}={eval_b['role_head']['accuracy_ref_holdout']}")

    common = set(pc_a) & set(pc_b)
    diffs = [pc_a[c] - pc_b[c] for c in common if pc_a[c] is not None and pc_b[c] is not None]
    forgetting = max([0.0] + diffs)  # same formula as backend/app/main.py's /versions/compare
    print(f"forgetting (max per-class control accuracy drop, floored at 0): {forgetting:.4f}")

    mil_a, mil_b = eval_a["military"], eval_b["military"]
    print(f"\nanalyst_class_assignment_rate: {BASE_VERSION_ID}={mil_a['analyst_class_assignment_rate']:.4f}  "
          f"{VERSION_ID}={mil_b['analyst_class_assignment_rate']:.4f}  (alias_note: {mil_b.get('alias_note')})")
    print(f"n_eval:                        {BASE_VERSION_ID}={mil_a['n_eval']}  {VERSION_ID}={mil_b['n_eval']}")
    print(f"n_excluded_used_in_fit:        {BASE_VERSION_ID}={eval_a['n_excluded_used_in_fit']}  "
          f"{VERSION_ID}={eval_b['n_excluded_used_in_fit']}")
    print(f"n_excluded_same_parent:        {BASE_VERSION_ID}={eval_a['n_excluded_same_parent']}  "
          f"{VERSION_ID}={eval_b['n_excluded_same_parent']}")


def _save_artifacts(feedback: list[dict]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (f"eval_{VERSION_ID}.json", f"{VERSION_ID}_model.json", f"{VERSION_ID}_batches.json"):
        shutil.copy(SCORES_DIR / name, ARTIFACT_DIR / name)
    (ARTIFACT_DIR / "feedback.json").write_text(json.dumps(feedback, indent=2))
    print(f"\nartifacts written to {ARTIFACT_DIR}")


def _cleanup() -> None:
    """Deletes every rehearsal_v2-specific file from data/scores/, and defensively removes any
    "rehearsal_v2" entry from versions.json (a no-op given the _save_version_entry monkeypatch in
    _build_scratch_version, but kept so this is provably true even if that monkeypatch is ever
    removed) -- so the live app's data/scores/ and versions.json end exactly as they started."""
    for name in (f"{VERSION_ID}.parquet", f"{VERSION_ID}_batches.json", f"{VERSION_ID}_model.json", f"eval_{VERSION_ID}.json"):
        p = SCORES_DIR / name
        if p.exists():
            p.unlink()
    versions_path = SCORES_DIR / "versions.json"
    versions = json.loads(versions_path.read_text())
    remaining = [v for v in versions if v["id"] != VERSION_ID]
    if remaining != versions:
        versions_path.write_text(json.dumps(remaining, indent=2))
    print(f"cleanup: {VERSION_ID} removed from data/scores/ and versions.json -- live app stays at "
          f"{BASE_VERSION_ID} only")


def main() -> None:
    feedback = _pick_feedback()
    n_new = sum(1 for f in feedback if f["action"] == "new_class")
    n_confirm = sum(1 for f in feedback if f["action"] == "confirm")
    print(f"feedback: {len(feedback)} items ({n_new} new_class '{NEW_LABEL}', {n_confirm} confirm)")
    print(f"  new_class ann_ids: {[f['ann_id'] for f in feedback if f['action'] == 'new_class']}")
    print(f"  confirm ann_ids:   {[f['ann_id'] for f in feedback if f['action'] == 'confirm']}")

    result = _build_scratch_version(feedback)
    eval_b = result["eval"]
    eval_a = json.loads((SCORES_DIR / f"eval_{BASE_VERSION_ID}.json").read_text())

    _print_compare(eval_a, eval_b)
    _save_artifacts(feedback)
    _cleanup()


if __name__ == "__main__":
    main()

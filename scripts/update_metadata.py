"""Update a Kaggle dataset's file and column descriptions, expected update frequency and
sources, without a new version and without touching anything else.

Kaggle's metadata update replaces the whole settings object (title, subtitle, description, tags,
licence, files), so this script first downloads the live metadata, changes only the fields named
in <dataset_dir>/settings.json, sends that back, then downloads again and checks every field.

settings.json:
  {"id": "owner/slug",
   "expectedUpdateFrequency": "...",           # optional
   "userSpecifiedSources": "...",              # optional
   "files": {"<file name>": {"description": "...", "columns": {"<column>": "<description>"}}}}

The file names and each file's column names must match the live dataset exactly, so a
description can never be attached to the wrong column or silently left out.

Usage: python scripts/update_metadata.py <dataset_dir> [--dry-run]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

FREQS = {"not specified", "never", "annually", "quarterly", "monthly", "weekly", "daily", "hourly"}
KEEP = ("title", "subtitle", "description", "keywords", "licenses", "isPrivate")


def fetch(api, ref: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = api.dataset_metadata(ref, d)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def merged(live: dict, want: dict) -> tuple[dict, list[str]]:
    """Return the metadata to send and a list of problems (empty when it is safe to send)."""
    problems = []
    live_files = {f["name"]: f for f in live.get("data") or []}
    want_files = want["files"]
    if set(live_files) != set(want_files):
        problems.append(f"file names differ: live {sorted(live_files)} vs settings {sorted(want_files)}")
    out = {k: live.get(k) for k in KEEP}
    if live.get("collaborators"):
        problems.append("the live dataset has collaborators; this script does not carry them over")
    data = []
    for name, lf in live_files.items():
        wf = want_files.get(name)
        if wf is None:
            continue
        live_cols = [c["name"] for c in lf.get("columns") or []]
        if set(live_cols) != set(wf["columns"]) or len(live_cols) != len(set(live_cols)):
            missing = sorted(set(live_cols) - set(wf["columns"]))
            extra = sorted(set(wf["columns"]) - set(live_cols))
            problems.append(f"{name}: columns differ (not described: {missing}; not in the file: {extra})")
        cols = []
        for c in lf.get("columns") or []:
            col = {"name": c["name"], "description": wf["columns"].get(c["name"], c.get("description") or "")}
            if c.get("type"):
                col["type"] = c["type"]
            cols.append(col)
        data.append({"name": name, "description": wf["description"], "columns": cols})
    out["data"] = data
    freq = want.get("expectedUpdateFrequency")
    if freq is not None:
        if freq not in FREQS:
            problems.append(f"expectedUpdateFrequency {freq!r} is not one of {sorted(FREQS)}")
        out["expectedUpdateFrequency"] = freq
    else:
        out["expectedUpdateFrequency"] = live.get("expectedUpdateFrequency")
    src = want.get("userSpecifiedSources")
    out["userSpecifiedSources"] = src if src is not None else (live.get("userSpecifiedSources") or "")
    return out, problems


def check(before: dict, after: dict, sent: dict) -> list[str]:
    bad = []
    for k in KEEP:
        if before.get(k) != after.get(k):
            bad.append(f"{k} changed: {before.get(k)!r} -> {after.get(k)!r}")
    for k in ("expectedUpdateFrequency", "userSpecifiedSources"):
        if (sent.get(k) or "") != (after.get(k) or ""):
            bad.append(f"{k}: sent {sent.get(k)!r}, now {after.get(k)!r}")
    now = {f["name"]: f for f in after.get("data") or []}
    for f in sent["data"]:
        g = now.get(f["name"])
        if g is None:
            bad.append(f"{f['name']} missing after the update")
            continue
        if (g.get("description") or "") != f["description"]:
            bad.append(f"{f['name']}: file description not stored")
        got = {c["name"]: c for c in g.get("columns") or []}
        for c in f["columns"]:
            h = got.get(c["name"], {})
            if (h.get("description") or "") != c["description"]:
                bad.append(f"{f['name']}.{c['name']}: description not stored")
            if c.get("type") and h.get("type") != c["type"]:
                bad.append(f"{f['name']}.{c['name']}: type changed {c['type']!r} -> {h.get('type')!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    want = json.loads((args.dataset_dir / "settings.json").read_text(encoding="utf-8"))
    ref = want["id"]

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    before = fetch(api, ref)
    print(f"live: {ref} usability {before.get('usabilityRating')} "
          f"freq {before.get('expectedUpdateFrequency')!r} sources {len(before.get('userSpecifiedSources') or '')} chars")
    for f in before.get("data") or []:
        described = sum(bool(c.get("description")) for c in f.get("columns") or [])
        print(f"  {f['name']}: file description {len(f.get('description') or '')} chars, "
              f"{described}/{len(f.get('columns') or [])} columns described")
    sent, problems = merged(before, want)
    if problems:
        print("NOT SENT:\n  " + "\n  ".join(problems))
        return 1
    n_cols = sum(len(f["columns"]) for f in sent["data"])
    print(f"ready: {len(sent['data'])} files, {n_cols} column descriptions, "
          f"freq {sent['expectedUpdateFrequency']!r}, sources {len(sent['userSpecifiedSources'])} chars")
    if args.dry_run:
        print("dry run: nothing sent")
        return 0

    with tempfile.TemporaryDirectory() as d:  # no cover image here, so the image is left as it is
        (Path(d) / "dataset-metadata.json").write_text(json.dumps(sent, ensure_ascii=False), encoding="utf-8")
        api.dataset_metadata_update(ref, d)

    after = fetch(api, ref)
    bad = check(before, after, sent)
    print(f"after: usability {after.get('usabilityRating')}")
    if bad:
        print("CHECK FAILED:\n  " + "\n  ".join(bad))
        return 1
    print("all fields stored as sent; nothing else changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

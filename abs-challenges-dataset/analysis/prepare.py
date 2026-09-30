"""Build the analysis table from challenges.jsonl (see PREREG.md). Standard library only."""
import json
import math
import random

HALF_PLATE = 17 / 2 / 12   # ft
BALL_R = 1.45 / 12         # ft
SEED = 20260929
N_EVAL = 250


def edge_in(px, pz, top, bot):
    """Signed distance (in) from the ball's surface to the zone; + = ball misses, - = touches.

    The set of ball centres whose ball touches the zone is the zone rectangle grown by the
    ball radius with ROUNDED corners, so: signed distance to the plain rectangle minus r.
    """
    x0, x1 = -HALF_PLATE, HALF_PLATE
    z0, z1 = bot, top
    dx = max(x0 - px, 0.0, px - x1)
    dz = max(z0 - pz, 0.0, pz - z1)
    if dx > 0 or dz > 0:
        d0 = math.hypot(dx, dz)
    else:
        d0 = -min(px - x0, x1 - px, pz - z0, z1 - pz)
    return (d0 - BALL_R) * 12


def load(path="challenges.jsonl"):
    rows, seen, dropped = [], set(), 0
    for line in open(path):
        r = json.loads(line)
        key = (r["gamePk"], r["inning"], r["half"], r["balls"], r["strikes"], r["pX"])
        if key in seen:          # a game re-fetched after a partial write
            continue
        seen.add(key)
        if None in (r["pX"], r["pZ"], r["sz_top"], r["sz_bot"], r["overturned"]):
            dropped += 1
            continue
        batting_id = r["home_id"] if r["batting_team_is_home"] else r["away_id"]
        if r["challenge_team"] not in (r["home_id"], r["away_id"]):
            dropped += 1
            continue
        r["challenged_by"] = "batting team" if r["challenge_team"] == batting_id else "fielding team"
        r["original_call"] = "strike" if r["challenged_by"] == "batting team" else "ball"
        e = edge_in(r["pX"], r["pZ"], r["sz_top"], r["sz_bot"])
        r["edge_in"] = e
        r["wrong_side_in"] = e if r["original_call"] == "strike" else -e
        r["y"] = 1 if r["overturned"] else 0
        rows.append(r)
    return rows, dropped


def split(rows):
    train = [r for r in rows if "2026-08-01" <= r["date"] <= "2026-08-31"]
    ev = [r for r in rows if "2026-09-01" <= r["date"] <= "2026-09-27"]
    ev.sort(key=lambda r: (r["gamePk"], r["inning"], r["half"], r["balls"], r["strikes"], r["pX"]))
    sample = random.Random(SEED).sample(ev, N_EVAL)
    return train, ev, sample


if __name__ == "__main__":
    rows, dropped = load()
    train, ev, sample = split(rows)
    # Consistency check of the derived original call against the post-review call:
    # overturned strike -> final Ball; overturned ball -> final Called Strike.
    bad = 0
    for r in rows:
        if r["call"] is None:
            bad += 1
            continue
        final_strike = "Strike" in r["call"]
        orig_strike = r["original_call"] == "strike"
        if (final_strike != orig_strike) != bool(r["y"]):
            bad += 1
    print(f"rows {len(rows)} dropped {dropped} train {len(train)} eval {len(ev)} sample {len(sample)}")
    print(f"overturn rate train {sum(r['y'] for r in train)/len(train):.3f} "
          f"eval {sum(r['y'] for r in ev)/len(ev):.3f} sample {sum(r['y'] for r in sample)/len(sample):.3f}")
    print(f"original-call derivation inconsistent with final call: {bad} rows")
    assert bad == 0, "original call derivation disagrees with the post-review call"
    json.dump(sample, open("eval_sample.json", "w"))
    json.dump(train, open("train.json", "w"))
    json.dump(ev, open("eval_all.json", "w"))

# %% [markdown]
# # Pre-freeze check: the after-outing notebook on the full files with random placements
#
# The draft notebook of the after-outing study, run with the frozen random-placement generator (no real placement is
# read or joined) so that its gates meet the full MLB 2024-2026 and Triple-A 2023-2026 files before freezing.

# %%
import os

os.environ["SMOKE_FAKE_LABELS"] = "1"  # random placements only

# %% [markdown]
# # Does the Last Outing Before the Injured List Look Different? (MLB pitchers, 2024-2026)
#
# Seven pre-registered pitcher studies ([FREEZE_pitch.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md) and its successors) built every feature of a pitcher-day from pitches thrown **strictly before** that day. The row of a pitcher's last appearance before a placement therefore never contained that appearance, while arm placements follow the last appearance within one day 73% of the time. This study scores **the outing that just ended**: its primary-fastball velocity against the pitcher's own recent normal.
#
# - **Target**: the last MLB appearance before an arm placement effective 1 to 14 days later (the other rows before that placement are left out); negatives are rows with no placement within 14 days.
# - **P1**: on outings without an early hook (left mid-at-bat, or a mid-inning exit 15 or more pitches short of his usual count), the AUC of the daily within-role rank of the velocity drop A. SUPPORTED if the bootstrap lower bound (0.625%) is above 0.5 **and** the rotation p-value (1,000 rotations from each pitcher-season's whole cyclic group) is at most 0.0125, and the same holds for A minus its position-in-season mean.
# - **P2**: MLB 2026, the model with A over a rival that also holds what anyone watching saw (start, pitch count, batters faced, pitches in the last inning, runs, mid-inning exit, left mid-at-bat, pitch-count shortfall).
#
# It runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) up to its tests, so rows, labels and the rival B0 are the frozen ones (equality gates). Everything was fixed before any outing-day feature met a placement: [FREEZE_pitch_after.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_after.md).

# %%
import hashlib

import requests

SHA = "2c6b01de48f5fd00dba18a5731404b40d4ddd707"
URL = f"https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/{SHA}/pitcher-anomaly-il/pitcher_anomaly_il.py"
src = requests.get(URL, timeout=60).text
assert hashlib.md5(src.encode()).hexdigest() == "100c853ccb8e42f386f547ead97f0b95", "not the frozen pitcher notebook"

import numpy
import pandas
import sklearn

print("versions: numpy", numpy.__version__, "| pandas", pandas.__version__, "| scikit-learn", sklearn.__version__)


def between(s, a, b):
    assert s.count(a) == 1 and s.count(b) == 1, (a[:40], s.count(a), b[:40], s.count(b))
    i = s.index(a)
    return s[i: s.index(b, i)]


HEAD = src[: src.index("# %% [markdown]\n# ## Tests")]
DEFS = between(src, "CTX = [", "\n\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\nP1 = ")
exec(compile(HEAD + "\n\n" + DEFS, "frozen_pitcher_up_to_tests", "exec"))

# %% [markdown]
# ## Equality gates: rows, labels and the rival are the frozen run's

# %%
if not FAKE:
    _got = (len(rows_m), rows_m.pitcher.nunique(), len(rows_a), rows_a.pitcher.nunique(), int(rows_m.use14.sum()),
            int(rows_m.il14[rows_m.use14].sum()), int(rows_a.use14.sum()), int(rows_a.il14[rows_a.use14].sum()))
    print("equality gates, rows and labels:", _got)
    assert _got == (61966, 1180, 55686, 1853, 53384, 3553, 45453, 1884), "rows or labels differ from the frozen run"
    _tr = (rows_m.season.isin(TRAIN) & rows_m.use14).to_numpy()
    _te = ((rows_m.season == TEST) & rows_m.use14).to_numpy()
    _b0 = auc(rows_m.il14[_te].to_numpy(), predict(fit(CTX, rows_m[_tr], rows_m.il14[_tr].to_numpy()), CTX, rows_m[_te]))
    print(f"equality gate, MLB 2026 AUC(B0): {_b0:.16f} (frozen run 0.5644912856858865)")
    assert abs(_b0 - 0.5644912856858865) < 1e-6, "the rival differs from the frozen run"

# %% [markdown]
# ## The outing of the row's own day (the only new input)

# %%
XCOLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "pitch_type",
         "release_speed", "game_type"]
XSCORE = ["bat_score", "post_bat_score"]
assert not [c for c in XCOLS + XSCORE if "until" in c or "next" in c], "future columns"
N_NORMAL, N_MIN_PITCHES, N_MIN_OWN, N_MIN_USUAL = 60, 5, 5, 3
LEAGUE = {"mlb": 2025, "aaa": 2023}


def load_ctx(pattern, years, cols, md5s):
    frames = []
    for y in years:
        p = glob.glob(os.path.join(INPUT_ROOT, "**", pattern.format(y=y)), recursive=True)
        assert len(p) == 1, (pattern.format(y=y), p)
        assert FAKE or hashlib.md5(open(p[0], "rb").read()).hexdigest() == md5s.get(y), "not the frozen file"
        d = pd.read_parquet(p[0], columns=cols)
        d["season"] = y
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out = out[out.game_type == "R"]  # every pitcher of the game: the next pitch decides an exit
    return out.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


ctx = {"mlb": load_ctx("statcast_{y}.parquet", ROW_SEASONS, XCOLS + XSCORE, MLB_MD5),
       "aaa": load_ctx("aaa_pitching_{y}.parquet", [2023] + ROW_SEASONS, XCOLS, AAA_MD5)}


def outing_items(d, level):
    """One record per pitcher, game and date (a suspended game resumed on a later date gives two): what anyone watching
    saw when he left (no later pitch of his that day)."""
    K = ["pitcher", "game_pk", "game_date"]
    gp, ab = d.game_pk.to_numpy(), d.at_bat_number.to_numpy()
    inn, tb, pid = d.inning.to_numpy(), d.inning_topbot.astype(str).to_numpy(), d.pitcher.to_numpy()
    nxt = np.r_[np.arange(1, len(d)), len(d) - 1]
    has_next = np.r_[gp[1:] == gp[:-1], False]
    last = d.groupby(K, sort=False).tail(1).index.to_numpy()  # sorted by game order: his last pitch
    j = nxt[last]
    hn = has_next[last]
    rec = pd.DataFrame({"pitcher": pid[last], "game_pk": gp[last], "day": d.game_date.to_numpy()[last],
                        "season": d.season.to_numpy()[last],
                        "o_mid_exit": hn & (inn[j] == inn[last]) & (tb[j] == tb[last]) & (pid[j] != pid[last]),
                        "o_left_mid_pa": hn & (ab[j] == ab[last]) & (pid[j] != pid[last])})
    g = d.groupby(K, sort=False)
    agg = pd.DataFrame({"o_pitches": g.size(), "o_batters": g.at_bat_number.nunique(), "o_first_inning": g.inning.min(),
                        "o_last_inning": g.inning.max()})
    fin = d.merge(agg[["o_last_inning"]], left_on=K, right_index=True)
    agg["o_final_inning"] = fin[fin.inning == fin.o_last_inning].groupby(K).size()
    if "bat_score" in d.columns:
        agg["o_runs"] = (d.post_bat_score - d.bat_score).groupby([d.pitcher, d.game_pk, d.game_date]).sum()
    else:
        agg["o_runs"] = np.nan
    rec = rec.merge(agg.reset_index().rename(columns={"game_date": "day"}), on=["pitcher", "game_pk", "day"], how="left",
                    validate="one_to_one")
    rec["o_start"] = (rec.o_first_inning == 1).astype(float)
    return rec


def vel_season(g, pm):
    """One pitcher-season at one level, sorted by date: the frozen primary-fastball choice per day, the mean velocity of
    that type on the day (x, at least 5 pitches) and of the last 60 of that type strictly before the day (normal)."""
    days = np.unique(g.game_date.to_numpy())
    gd = g.game_date.to_numpy()
    pt = g.pitch_type.fillna("").to_numpy(dtype=object)
    before = np.vstack([np.searchsorted(gd[pt == t], days, side="left") for t in FB])
    choice = np.where(before.max(0) > 0, before.argmax(0), FB.index(pm) if pm else -1)
    x = np.full(len(days), np.nan)
    normal = np.full(len(days), np.nan)
    fbv_long = np.full(len(days), np.nan)
    for ti, t in enumerate(FB):
        m = choice == ti
        if not m.any():
            continue
        s_ = g[pt == t]
        sd_, v_ = s_.game_date.to_numpy(), s_.release_speed.to_numpy(float)
        normal[m] = wmean(sd_, v_, N_NORMAL, days[m])
        fbv_long[m] = wmean(sd_, v_, LONG[0], days[m])
        lo_, hi_ = np.searchsorted(sd_, days[m], side="left"), np.searchsorted(sd_, days[m], side="right")
        x[m] = [np.nanmean(v_[a:z]) if np.isfinite(v_[a:z]).sum() >= N_MIN_PITCHES else np.nan for a, z in zip(lo_, hi_)]
    return pd.DataFrame({"day": days, "choice": choice, "x": x, "normal": normal, "fbv_long": fbv_long})


def usual_of(pc, st):
    """Median of his earlier single-game outings this season with the same start flag (at least 3, strictly before)."""
    usual = np.full(len(pc), np.nan)
    for k in range(len(pc)):
        if not np.isfinite(st[k]):
            continue
        prior = pc[:k][(st[:k] == st[k]) & np.isfinite(pc[:k])]
        if len(prior) >= N_MIN_USUAL:
            usual[k] = np.median(prior)
    return usual


def own_scale_of(dv):
    """MAD of the earlier dev values of the pitcher-season (at least 5, strictly before the day)."""
    own = np.full(len(dv), np.nan)
    for k in range(len(dv)):
        prior = dv[:k][np.isfinite(dv[:k])]
        if len(prior) >= N_MIN_OWN:
            own[k] = mad(prior)
    return own


def look_ahead_gate(groups, modes, level):
    """Pitches after a day t0 set to 999 move nothing up to t0; pitches on t0 set to 999 move x at t0 but not the
    normal at t0, nor the own scale at t0."""
    found = None
    for k0 in sorted(groups, key=lambda k: -len(groups[k])):
        g0 = groups[k0]
        base = vel_season(g0, modes.get((k0[0], k0[1] - 1)))
        dd_, xx, nn = base.day.to_numpy(), base.x.to_numpy(), base.normal.to_numpy()
        nxt_day = np.r_[(dd_[1:] - dd_[:-1]) == np.timedelta64(1, "D"), False]
        ok_ = np.flatnonzero(nxt_day & np.isfinite(xx) & np.isfinite(nn) & np.r_[np.isfinite(xx[1:]), False])
        if len(ok_):
            found = (k0, g0, base, int(ok_[len(ok_) // 2]))
            break
    assert found is not None, "no day followed by an appearance the next day"
    k0, g0, base, i0 = found
    t0 = base.day.to_numpy()[i0]
    later = g0.copy()
    later.loc[(later.game_date > t0).to_numpy(), "release_speed"] = 999.0
    later.loc[(later.game_date > t0).to_numpy(), "pitch_type"] = "FC"
    v1 = vel_season(later, modes.get((k0[0], k0[1] - 1)))
    for c in ("choice", "x", "normal"):
        assert np.array_equal(base[c].to_numpy()[: i0 + 1], v1[c].to_numpy()[: i0 + 1], equal_nan=True), f"look-ahead in {c}"
    same = g0.copy()
    same.loc[(same.game_date == t0).to_numpy(), "release_speed"] = 999.0
    v2 = vel_season(same, modes.get((k0[0], k0[1] - 1)))
    assert v2.x.to_numpy()[i0] != base.x.to_numpy()[i0], "x does not read the day's own pitches"
    assert v2.normal.to_numpy()[i0] == base.normal.to_numpy()[i0], "the normal reads the day's own pitches"
    d0, d2 = (base.x - base.normal).to_numpy(), (v2.x - v2.normal).to_numpy()
    o0, o2 = own_scale_of(d0), own_scale_of(d2)
    assert np.array_equal(o0[: i0 + 1], o2[: i0 + 1], equal_nan=True), "the own scale reads the day's own dev"
    print(f"{level}: look-ahead gates passed on pitcher {k0[0]} season {k0[1]} at day {str(t0)[:10]}")


def exit_selftest():
    """A synthetic game (home pitchers in the top halves, X for the away team in the bottom halves): A leaves between
    innings; B leaves in the middle of at-bat 4 (C throws its last pitch); C leaves between innings; D leaves mid-inning
    after a completed at-bat; F leaves between innings; X throws the game's last pitch."""
    rows_ = []
    seq = [(1, "Top", "A", 1), (1, "Top", "A", 2), (1, "Bot", "X", 3), (2, "Top", "B", 4), (2, "Top", "C", 5),
           (2, "Bot", "X", 6), (3, "Top", "D", 7), (3, "Top", "F", 8), (3, "Bot", "X", 9)]
    for inn, tb, who, ab in seq:
        for pn in (1, 2):
            rows_.append({"pitcher": who, "game_pk": 1, "game_date": pd.Timestamp("2025-05-01"), "season": 2025,
                          "inning": inn, "inning_topbot": tb, "at_bat_number": ab, "pitch_number": pn,
                          "bat_score": 0, "post_bat_score": 0})
    # C enters in the middle of B's at-bat 4: one more pitch of at-bat 4 thrown by C
    rows_.insert(8, {"pitcher": "C", "game_pk": 1, "game_date": pd.Timestamp("2025-05-01"), "season": 2025, "inning": 2,
                     "inning_topbot": "Top", "at_bat_number": 4, "pitch_number": 3, "bat_score": 0, "post_bat_score": 0})
    d = pd.DataFrame(rows_).sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    o = outing_items(d, "selftest").set_index("pitcher")
    want = {"A": (False, False), "B": (True, True), "C": (False, False), "D": (True, False), "F": (False, False),
            "X": (False, False)}
    for who, (mid, mpa) in want.items():
        assert (bool(o.o_mid_exit[who]), bool(o.o_left_mid_pa[who])) == (mid, mpa), ("exit flags wrong", who, o.loc[who].to_dict())
    print("exit self-test passed")


exit_selftest()


def day_records(d, level):
    """One record per pitcher and appearance day: the visible items and the velocity drop A of the day's outing."""
    rec = outing_items(d, level)
    n_games = rec.groupby(["pitcher", "day"]).game_pk.transform("size")
    rec = rec.sort_values(["pitcher", "day", "game_pk"], kind="stable")
    one = rec.drop_duplicates(["pitcher", "day"]).copy()
    one["double"] = (n_games.reindex(one.index) > 1).to_numpy()
    for c in ["o_mid_exit", "o_left_mid_pa", "o_pitches", "o_batters", "o_final_inning", "o_runs", "o_start"]:
        one[c] = np.where(one.double, np.nan, one[c].astype(float))
    one = one.sort_values(["pitcher", "season", "day"], kind="stable").reset_index(drop=True)
    # usual pitch count: median of his earlier single-game outings this season with the same start flag (at least 3)
    usual = np.full(len(one), np.nan)
    gi = list(one.groupby(["pitcher", "season"]).indices.values())
    for ix in gi:
        usual[ix] = usual_of(one.o_pitches.to_numpy()[ix], one.o_start.to_numpy()[ix])
    big = max(gi, key=len)  # gate: a huge pitch count on one day does not move that day's usual count
    pc_, st_ = one.o_pitches.to_numpy()[big].copy(), one.o_start.to_numpy()[big]
    i0 = len(big) // 2
    pc_[i0] = 9999.0
    assert np.array_equal(usual_of(pc_, st_)[: i0 + 1], usual[big][: i0 + 1], equal_nan=True), "the usual pitch count reads the day itself"
    lg = one[one.season.eq(LEAGUE[level])].groupby("o_start").o_pitches.median()
    fallback = one.o_start.map(lg).to_numpy(float)
    one["o_shortfall"] = np.where(np.isfinite(usual), usual, fallback) - one.o_pitches
    hook = np.where(one.o_left_mid_pa == 1, 1.0, np.where(one.o_mid_exit == 1,
                    np.where(np.isfinite(one.o_shortfall), (one.o_shortfall >= 15).astype(float), np.nan), 0.0))
    one["early_hook"] = np.where(one.double, np.nan, hook)
    # velocity of the day's primary fastball (the frozen choice) against the last 60 of that type before the day
    dd = d.sort_values(["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
    groups = {k: g for k, g in dd.groupby(["pitcher", "season"], sort=False)}
    modes = {}
    for k, g in groups.items():
        c = g.pitch_type.value_counts()
        cnt = [c.get(t, 0) for t in FB]
        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None
    out = [vel_season(g, modes.get((b, y - 1))).assign(pitcher=b, season=y) for (b, y), g in groups.items()]
    vel = pd.concat(out, ignore_index=True)
    one = one.merge(vel, on=["pitcher", "season", "day"], how="left", validate="one_to_one")
    one["dev"] = np.where(one.double, np.nan, one.x - one.normal)
    own = np.full(len(one), np.nan)
    for _, ix in one.groupby(["pitcher", "season"]).indices.items():
        own[ix] = own_scale_of(one.dev.to_numpy()[ix])
    one["own_scale"] = own
    look_ahead_gate(groups, modes, level)
    srole = one.groupby(["pitcher", "season"]).o_start.transform(lambda v: float(np.nanmean(v) > 0.5))
    lsc = {}
    for role in (0.0, 1.0):
        ms = [mad(v[np.isfinite(v)]) for (b, y), v in one[srole.eq(role)].groupby(["pitcher", "season"]).dev
              if y == LEAGUE[level] and np.isfinite(v.to_numpy()).sum() >= MIN_SCALE]
        lsc[role] = float(np.median(ms)) if ms else np.nan
    print(f"{level}: league-typical scale of dev (season {LEAGUE[level]}): relievers {lsc[0.0]:.3f}, starters {lsc[1.0]:.3f} mph")
    assert all(np.isfinite(v) and v > 0 for v in lsc.values()), "a league-typical scale is missing"
    return one, lsc


def attach_after(rows, level):
    one, lsc = day_records(ctx[level], level)
    r = rows.merge(one.drop(columns=["game_pk"]), on=["pitcher", "season", "day"], how="left", validate="one_to_one",
                   indicator=True)
    assert len(r) == len(rows) and r._merge.eq("both").all(), f"{level}: rows without an outing record"
    r = r.drop(columns="_merge")
    agree = ((r.choice < 0) == r.no_primary).mean()
    same_long = np.isclose(r.fbv_long, r["FB velocity"], rtol=0, atol=1e-9, equal_nan=True).mean()
    print(f"{level}: primary-fastball choice agrees with the frozen no_primary flag on {agree:.4%} of rows; the chosen "
          f"type's 150-pitch mean equals the frozen FB velocity on {same_long:.4%}")
    if level == "mlb":
        assert agree == 1.0 and same_long == 1.0, "the primary-fastball choice is not the frozen one"
    d = ctx[level]
    ch = r[["pitcher", "season", "day", "choice"]][r.choice >= 0].assign(T=lambda q: [FB[c] for c in q.choice])
    pp = d.merge(ch, left_on=["pitcher", "season", "game_date"], right_on=["pitcher", "season", "day"])
    pp = pp[pp.pitch_type == pp["T"]]
    gx = pp.groupby(["pitcher", "season", "day"]).release_speed.agg(["mean", "count"])
    x_ind = gx["mean"].where(gx["count"] >= N_MIN_PITCHES).rename("x_ind").reset_index()
    cmp_ = r[["pitcher", "season", "day", "x"]].merge(x_ind, on=["pitcher", "season", "day"], how="left")
    assert np.allclose(cmp_.x, cmp_.x_ind, rtol=0, atol=1e-9, equal_nan=True), "x is not the mean of the day's chosen-type pitches"
    rng_n = np.random.default_rng(11)
    by_ps = {k: g.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
             for k, g in d.groupby(["pitcher", "season"])}
    chk = r.index[np.isfinite(r.normal)].to_numpy()
    for i in rng_n.choice(chk, min(200, len(chk)), replace=False):
        b, y, dy, c = r.pitcher[i], r.season[i], r.day[i], FB[int(r.choice[i])]
        g = by_ps[(b, y)]
        v = g.release_speed[(g.pitch_type == c) & (g.game_date < dy)].to_numpy(float)[-N_NORMAL:]
        want = np.nanmean(v) if len(v) == N_NORMAL and np.isfinite(v).sum() >= N_NORMAL / 2 else np.nan
        assert np.isclose(want, r.normal[i], rtol=0, atol=1e-9, equal_nan=True), ("normal differs from a direct loop", b, y, dy)
    single = ~r.double.astype(bool)
    prev_ok = (r.pitcher.eq(r.pitcher.shift()) & r.season.eq(r.season.shift()) & single & single.shift(fill_value=False)).to_numpy()
    pa_ok = np.isclose(r.prev_app_pitches.to_numpy()[prev_ok], r.o_pitches.shift().to_numpy()[prev_ok]).mean()
    st_ok = np.isclose(r.prev_start.to_numpy()[prev_ok], r.o_start.shift().to_numpy()[prev_ok]).mean()
    print(f"{level}: previous outing's pitches / start flag equal the frozen rows' on {pa_ok:.4%} / {st_ok:.4%} of rows")
    if level == "mlb":
        assert pa_ok == 1.0 and st_ok == 1.0, "outing items disagree with the frozen rows"
    vis = ["o_mid_exit", "o_left_mid_pa", "o_pitches", "o_batters", "o_final_inning", "o_runs", "o_start", "o_shortfall", "early_hook"]
    assert not np.isfinite(r.loc[r.double.astype(bool), vis].to_numpy(float)).any(), "outing items defined on a doubleheader day"
    ls = np.where(r.role.eq(1.0), lsc[1.0], lsc[0.0])
    scale = np.where(np.isfinite(r.own_scale), np.maximum(r.own_scale, 0.5 * ls), ls)
    r["A"] = np.clip(-(r.dev / scale), -10, 10)
    r["A_rank"] = daily_rank(r, "A")
    r["A2_rank"] = daily_rank(r.assign(A2=r.A ** 2), "A2")
    print(f"{level}: doubleheader days {int(r.double.sum()):,}; A defined on {np.isfinite(r.A).mean():.1%} of rows "
          f"(starters {np.isfinite(r.A[r.role.eq(1.0)]).mean():.1%}, relievers {np.isfinite(r.A[r.role.eq(0.0)]).mean():.1%}); "
          f"mid-inning exit {np.nanmean(r.o_mid_exit):.3f}, left mid-at-bat {np.nanmean(r.o_left_mid_pa):.4f}, "
          f"early hook {np.nanmean(r.early_hook):.3f}")
    if not FAKE and level == "mlb":  # the exit flags must vary (an always-true or never-true flag is a definition error)
        for role in (0.0, 1.0):
            rate = np.nanmean(r.o_mid_exit[r.role.eq(role)])
            assert 0.05 < rate < 0.95, ("mid-inning exit rate implausible", role, rate)
        assert 0.0 < np.nanmean(r.o_left_mid_pa) < 0.05, "left mid-at-bat rate implausible"
    return r


rows_m = attach_after(rows_m, "mlb")
rows_a = attach_after(rows_a, "aaa")

# %% [markdown]
# ## The target: the last appearance before an arm placement 1 to 14 days later

# %%


def last_outing_target(r, name):
    """1 = the last row before an arm placement effective 1-14 days after it (usable); 0 = usable row with no placement
    within 14 days; NaN otherwise (earlier rows before the same placement, gap 0, non-arm placements, unusable rows)."""
    y = np.where(r.use14 & ~r.il14, 0.0, np.nan)
    il = r[r.il14].copy()
    il["pl"] = il.day + pd.to_timedelta(il.days_to_il, unit="D")
    last = il.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
    keep = last[last.next_arm & last.days_to_il.ge(1) & last.use14]
    y[keep.index.to_numpy()] = 1.0
    n_arm = int(last.next_arm.sum())
    print(f"{name}: arm placements with a row in the 14 days before {n_arm:,}; positives {len(keep):,}; left out: gap 0 "
          f"{int((last.next_arm & last.days_to_il.eq(0)).sum()):,}, last row unusable "
          f"{int((last.next_arm & last.days_to_il.ge(1) & ~last.use14).sum()):,}; non-arm placements {int((~last.next_arm).sum()):,}")
    return y


rows_m["y_last"] = last_outing_target(rows_m, "MLB")
rows_a["y_last"] = last_outing_target(rows_a, "Triple-A")
for _nm, _r in (("MLB", rows_m), ("Triple-A", rows_a)):
    _p = _r[_r.y_last.eq(1.0)]
    assert _p.days_to_il.between(1, H).all(), f"{_nm}: a positive outside 1 to {H} days before its placement"
    _days = {b: np.sort(g.day.to_numpy()) for b, g in _r.groupby("pitcher")}
    _later = [((v := _days[b]) > d) & (v <= d + np.timedelta64(int(k), "D")) for b, d, k in zip(_p.pitcher, _p.day.to_numpy(), _p.days_to_il)]
    assert not any(m.any() for m in _later), f"{_nm}: a positive is not the last appearance before its placement"
    assert not np.isfinite(_r.A[_r.double.eq(True)]).any(), f"{_nm}: A defined on a doubleheader day"
    _lv = "mlb" if _nm == "MLB" else "aaa"
    _gl = [((v := app_days[_lv].get(b, EMPTY)) > d) & (v <= d + np.timedelta64(int(k), "D"))
           for b, d, k in zip(_p.pitcher, _p.day.to_numpy(), _p.days_to_il)]
    _share = np.mean([m.any() for m in _gl]) if len(_gl) else 0.0
    print(f"{_nm}: positives with a StatsAPI game-log appearance before their placement {_share:.2%}")
    assert _share <= 0.02, f"{_nm}: the game logs contradict too many positives"


def position_adjusted(r, col):
    """The score minus its mean by role and decile of the day of the season (known on the day; no label)."""
    dec = r.groupby("role").day_of_season.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
    means = r.groupby([r.role, dec])[col].transform("mean")
    print(f"  label-free mean of {col} by position decile (relievers / starters):\n"
          + r.groupby([r.role, dec])[col].mean().unstack(0).round(4).to_string())
    return r[col] - means


rows_m["A_rank_pos"] = position_adjusted(rows_m, "A_rank")

# %% [markdown]
# ## P1: does the last outing before an arm placement look slow, among outings without an early hook?

# %%
N_ROT = 1000


def rotated_group(vals, groups, sd):
    """Each pitcher-season with at least 7 values is rolled by k drawn uniformly from 0..n-1 (shift 0 included)."""
    rng = np.random.default_rng(sd)
    rot = vals.copy()
    for f in groups:
        if len(f) >= 7:
            rot[f] = np.roll(vals[f], int(rng.integers(0, len(f))))
    return rot


def floor_structure(df, gix, s0):
    assert np.array_equal(np.sort(np.concatenate(gix)), np.arange(len(df))), "groups do not cover the rows once"
    key, days = df[["pitcher", "season"]].to_numpy(), df.day.to_numpy()
    for f in gix:
        assert (key[f] == key[f][0]).all() and (np.diff(days[f]) > np.timedelta64(0, "D")).all(), \
            "groups are not pitcher-seasons in time order"
    ks, out = [], []
    for sd in range(s0, s0 + 50):
        ri = rotated_group(np.arange(len(df), dtype=float), gix, sd).astype(int)
        for f in gix:
            rr = ri[f]
            if len(f) < 7:
                assert np.array_equal(rr, f), "a pitcher-season with fewer than 7 values was rotated"
                continue
            k = (-int(np.flatnonzero(f == rr[0])[0])) % len(f)
            assert np.array_equal(rr, np.roll(f, k)), "the floor is not a rotation within the pitcher-season"
            m = max(2, int(np.ceil(len(f) / 6)))
            ks.append(k)
            out.append(k < m or k > len(f) - m)
    assert 0 in ks, "shift 0 is never drawn"
    assert np.mean(out) > 0.2, "the shifts avoid the band near 0"


def rotation_test(r, mask, score, name, s0, alpha, gate=False):
    df = r[mask & r.y_last.notna() & np.isfinite(r[score])].reset_index(drop=True)
    y, s = df.y_last.to_numpy() == 1.0, df[score].to_numpy()
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    a = auc(y, s)
    fl = np.array([auc(y, rotated_group(s, gix, sd)) for sd in range(s0, s0 + N_ROT)])
    p = (1 + int((fl >= a).sum())) / (N_ROT + 1)
    if gate:
        floor_structure(df, gix, s0)
        for i in range(3):
            ri = rotated_group(np.arange(len(df), dtype=float), gix, s0 + i).astype(int)
            assert fl[i] == auc(y, s[ri]), "the floor is not the whole-group rotation"
        if FAKE:  # the floor must be beatable by a planted signal
            pl = np.clip(s + 0.3 * y, 0, 1.3)
            pf = np.array([auc(y, rotated_group(pl, gix, sd)) for sd in range(s0, s0 + N_ROT)])
            assert (1 + int((pf >= auc(y, pl)).sum())) / (N_ROT + 1) <= alpha, f"{name}: the floor cannot be beaten by a planted signal"
    b = boot(df, y, {"A": s}, [])
    lo = b.loc["A", "lo_bonf"]
    ok = bool(lo > 0.5 and p <= alpha)
    print(f"{name} [{score}]: AUC {a:.4f} [{lo:.4f}, {b.loc['A', 'hi_bonf']:.4f}]; whole-group floor mean {fl.mean():.4f}, "
          f"97.5th percentile {np.percentile(fl, 97.5):.4f}, p {p:.4f}; positives {int(y.sum()):,}, rows {len(df):,}, "
          f"pitchers {df.pitcher.nunique():,} -> {'passes' if ok else 'does not pass'}")
    return {"auc": a, "lo": lo, "hi": b.loc["A", "hi_bonf"], "p": p, "floor": fl, "ok": ok, "pos": int(y.sum()), "df": df}


ALPHA = 0.0125
no_hook = rows_m.early_hook.eq(0.0).to_numpy()
P1 = rotation_test(rows_m, no_hook, "A_rank", "P1 MLB, last outing, no early hook", 7000, ALPHA, gate=True)
P1P = rotation_test(rows_m, no_hook, "A_rank_pos", "P1 position-adjusted", 7000, ALPHA)
p1_ok = bool(P1["ok"] and P1P["ok"])
print(f"P1: {'SUPPORTED' if p1_ok else 'NOT SUPPORTED'} (both lines must pass)")

# Reported, not tested
R_RAW = rotation_test(rows_m, np.ones(len(rows_m), bool), "A_rank", "reported: all outings (manipulation check)", 7000, ALPHA)
R_A2 = rotation_test(rows_m, no_hook, "A2_rank", "reported: two-sided A^2", 7000, ALPHA)
_il = rows_m.assign(y_last=np.where(rows_m.use14 & (~rows_m.il14 | rows_m.next_arm), rows_m.il14.astype(float), np.nan))
R_IL14 = rotation_test(_il, np.ones(len(_il), bool), "A_rank", "reported: IL14 over all rows (frozen arm target)", 7000, ALPHA)
R_SUB = {}
for nm, m in [(f"season {y}", rows_m.season.eq(y)) for y in ROW_SEASONS] + [("starters", rows_m.role.eq(1.0)), ("relievers", rows_m.role.eq(0.0))]:
    R_SUB[nm] = rotation_test(rows_m, no_hook & m.to_numpy(), "A_rank", f"reported subset {nm}", 7000, ALPHA)
_pos = rows_m.y_last.eq(1.0)
print("early hooks among positives / negatives:", round(float(np.nanmean(rows_m.early_hook[_pos])), 4),
      "/", round(float(np.nanmean(rows_m.early_hook[rows_m.y_last.eq(0.0)])), 4))
R_AAA = rotation_test(rows_a, rows_a.early_hook.eq(0.0).to_numpy(), "A_rank", "reported: Triple-A, last outing, no early hook", 9000, ALPHA)

# %% [markdown]
# ## P2: MLB 2026, the velocity drop over a rival that holds what anyone watching saw

# %%
VIS = ["o_start", "o_pitches", "o_batters", "o_final_inning", "o_runs", "o_mid_exit", "o_left_mid_pa", "o_shortfall"]
SETS_A = {"B0+": CTX + VIS, "M+": CTX + VIS + ["A", "dev"]}
N_FLOOR_A = 80


def rotate_a(r, sd):
    out = r.copy()
    gix = list(out.groupby(["pitcher", "season"]).indices.values())
    out["A"] = rotated_group(out.A.to_numpy(), gix, sd)
    out["dev"] = rotated_group(out.dev.to_numpy(), gix, sd)
    return out


def model_test_a(r, name):
    tr = (r.season.isin(TRAIN) & r.y_last.notna()).to_numpy()
    te = ((r.season == TEST) & r.y_last.notna()).to_numpy()
    y_tr, y_te = r.y_last[tr].to_numpy() == 1.0, r.y_last[te].to_numpy() == 1.0
    models = {k: fit(c, r[tr], y_tr) for k, c in SETS_A.items()}
    pred = {k: predict(models[k], c, r[te]) for k, c in SETS_A.items()}
    test = r[te].reset_index(drop=True)
    main = boot(test, y_te, pred, [("M+", "B0+")])
    gap = main.loc["M+ - B0+", "AUC"]
    floor = []
    for i in range(N_FLOOR_A):
        rr = rotate_a(r, 8000 + i)
        floor.append(auc(y_te, predict(fit(SETS_A["M+"], rr[tr], y_tr), SETS_A["M+"], rr[te])) - main.loc["B0+", "AUC"])
    floor = np.array(floor)
    ok = bool(main.loc["M+ - B0+", "lo_bonf"] > 0 and np.all(gap > floor))
    print(f"{name}\n" + main.round(4).to_string())
    print(f"{name}: M+ - B0+ {gap:+.4f} [{main.loc['M+ - B0+', 'lo_bonf']:+.4f}, {main.loc['M+ - B0+', 'hi_bonf']:+.4f}]; "
          f"floor ({N_FLOOR_A}) mean {floor.mean():+.4f}, max {floor.max():+.4f}; positives {int(y_te.sum()):,} -> "
          f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    if FAKE:  # the floor must be beatable by a planted timing signal in A
        rp = r.copy()
        rp["A"] = np.where(np.isfinite(rp.A), np.where(rp.y_last.eq(1.0), 4.0, 0.0), np.nan)
        b_ = auc(y_te, predict(fit(SETS_A["B0+"], rp[tr], y_tr), SETS_A["B0+"], rp[te]))
        real = auc(y_te, predict(fit(SETS_A["M+"], rp[tr], y_tr), SETS_A["M+"], rp[te])) - b_
        fl = [auc(y_te, predict(fit(SETS_A["M+"], q[tr], y_tr), SETS_A["M+"], q[te])) - b_
              for q in (rotate_a(rp, 8000 + i) for i in range(10))]
        print(f"planted timing in A: real {real:+.4f}, floors max {max(fl):+.4f}")
        assert real > max(fl), "P2's floor cannot be beaten by planted timing"
    imp = permutation_importance(models["M+"], test[SETS_A["M+"]].to_numpy(float), y_te, scoring="roc_auc", n_repeats=5,
                                 random_state=0)
    print(f"{name} permutation importance (M+), top 10:\n"
          + pd.Series(imp.importances_mean, index=SETS_A["M+"]).sort_values(ascending=False).head(10).round(4).to_string())
    return {"main": main, "gap": gap, "floor": floor, "ok": ok, "pos": int(y_te.sum())}


P2 = model_test_a(rows_m, "P2 MLB 2026")

# %% [markdown]
# ## Verdicts and figure

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(np.zeros(len(P1["floor"])), P1["floor"], color="#bbbbbb", s=12, label="rotations (whole group)")
ax.scatter([0], [P1["auc"]], color="#c0392b", s=110, zorder=3, label="real")
ax.axhline(0.5, color="#cccccc", lw=1)
ax.axhline(np.percentile(P1["floor"], 100 - 100 * ALPHA), color="#888888", lw=1, ls="--", label="98.75th percentile")
ax.set_xticks([0], ["P1 MLB 2024-2026, last outing"])
ax.set_ylabel("AUC")
ax.set_title("Velocity drop in the last outing before an arm placement")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("after_floor.png", dpi=120)
plt.show()

summary = pd.DataFrame([
    {"test": "P1 MLB 2024-2026 last outing before an arm placement, no early hook, daily rank of A", "value": P1["auc"],
     "lower": P1["lo"], "p_rotation": P1["p"], "position_adjusted": P1P["auc"], "position_adjusted_lower": P1P["lo"],
     "position_adjusted_p": P1P["p"], "positives": P1["pos"], "verdict": "SUPPORTED" if p1_ok else "NOT SUPPORTED"},
    {"test": "P2 MLB 2026 AUC(M+) - AUC(B0+)", "value": P2["gap"], "lower": P2["main"].loc["M+ - B0+", "lo_bonf"],
     "p_rotation": np.nan, "position_adjusted": np.nan, "position_adjusted_lower": np.nan, "position_adjusted_p": np.nan,
     "positives": P2["pos"], "verdict": "SUPPORTED" if P2["ok"] else "NOT SUPPORTED"}])
if FAKE:
    print("FAKE LABELS: code test only, no verdict written")
else:
    summary.to_csv("after_verdict.csv", index=False)
print(summary.to_string(index=False))

"""Why was a driver dominant? Factor breakdown per driver-season.

Factors
    car          car's rank on the grid and gap to the next-best other car
    driver       skill rank on the grid and teammate head-to-head
    competition  strongest rival "package" (driver skill + car) vs own
    reliability  own car-failure rate vs the top rivals'
    new rules    major rule change this year or last (data/regulations.csv),
                 and how much the team's car improved over the year before
Driver/car split: driver edge and car edge over the season's average starter,
both in the same model units, so their shares can be compared.

Labels are comparative: thresholds are taken from title-winning seasons, so
"big margin" means the same thing in every era.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_STARTS = 5
TOP_RIVALS = 5

NUMERIC = [
    "driver_edge", "car_edge", "driver_share", "car_rank", "car_gap", "skill_rank", "h2h",
    "package_gap", "own_fail", "rivals_fail", "car_jump", "starts", "wins",
]


def load_regulations(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["year", "kind", "change"])


def _mode(s: pd.Series) -> object:
    m = s.dropna().mode()
    return m.iat[0] if len(m) else np.nan


def driver_share(driver_edge: pd.Series, car_edge: pd.Series) -> pd.Series:
    """Share of the advantage from the driver; NaN when neither is above average."""
    d, c = driver_edge.clip(lower=0), car_edge.clip(lower=0)
    total = d + c
    return (d / total).where(total > 0)


def season_factors(entries: pd.DataFrame, seasons: pd.DataFrame, cars: pd.DataFrame, regs: pd.DataFrame) -> pd.DataFrame:
    """One row per driver-season with every numeric factor."""
    s = seasons[["driver_id", "year", "starts", "wins", "rating", "car", "h2h", "champ_rank"]].copy()
    s["package"] = s["rating"] + s["car"]

    # Edges over the season's average starter (start-weighted), so eras compare.
    w = s["starts"]
    for col, out in (("rating", "driver_edge"), ("car", "car_edge")):
        avg = (s[col] * w).groupby(s["year"]).sum() / w.groupby(s["year"]).sum()
        s[out] = s[col] - s["year"].map(avg)
    s["driver_share"] = driver_share(s["driver_edge"], s["car_edge"])

    # Car rank and gap to the best car the driver did not drive.
    car_by_ts = cars.set_index("team_season")["car"]
    ts_year = entries.drop_duplicates("team_season").set_index("team_season")["year"]
    grid_cars = pd.DataFrame({"car": car_by_ts.reindex(ts_year.index), "year": ts_year}).dropna()
    own_ts = entries.groupby(["driver_id", "year"])["team_season"].agg(set)
    ranks, gaps = [], []
    for d, y, own in zip(s["driver_id"], s["year"], s["car"]):
        yr = grid_cars.loc[grid_cars["year"] == y, "car"]
        ranks.append(1 + int((yr > own + 1e-9).sum()))
        others = yr.drop(list(own_ts.get((d, y), set())), errors="ignore")
        gaps.append(own - others.max() if len(others) else np.nan)
    s["car_rank"], s["car_gap"] = ranks, gaps

    # Skill rank and rival among regular drivers that season.
    regular = s.loc[s["starts"] >= MIN_STARTS]
    s["skill_rank"] = s.apply(
        lambda r: 1 + int((regular.loc[regular["year"] == r["year"], "rating"] > r["rating"] + 1e-9).sum()), axis=1
    )

    fails = (entries["status"] == "car").groupby([entries["driver_id"], entries["year"]]).mean()
    s["own_fail"] = [fails.get((d, y), np.nan) for d, y in zip(s["driver_id"], s["year"])]
    rival_rows = []
    for d, y in zip(s["driver_id"], s["year"]):
        others = regular.loc[(regular["year"] == y) & (regular["driver_id"] != d)].nlargest(TOP_RIVALS, "package")
        if others.empty:
            rival_rows.append((np.nan, np.nan, np.nan, False))
            continue
        top = others.iloc[0]
        same_car = bool(own_ts.get((d, y), set()) & own_ts.get((top["driver_id"], y), set()))
        rivals_fail = np.nanmean([fails.get((r, y), np.nan) for r in others["driver_id"]])
        rival_rows.append((top["driver_id"], top["package"], rivals_fail, same_car))
    s[["rival_id", "rival_package", "rivals_fail", "rival_same_car"]] = pd.DataFrame(rival_rows, index=s.index)
    s["package_gap"] = s["package"] - s["rival_package"].astype(float)

    # Model-free car clue: where the best regular teammate finished in the championship.
    rank_of = s.set_index(["driver_id", "year"])["champ_rank"]
    starts_of = s.set_index(["driver_id", "year"])["starts"]
    ts_drivers = entries.groupby("team_season")["driver_id"].agg(set)
    tm_rows = []
    for d, y in zip(s["driver_id"], s["year"]):
        mates = set().union(*(ts_drivers[t] for t in own_ts.get((d, y), set()))) - {d}
        mates = [m for m in mates if starts_of.get((m, y), 0) >= MIN_STARTS]
        best = min(mates, key=lambda m: rank_of[(m, y)], default=None)
        tm_rows.append((best, rank_of[(best, y)] if best else np.nan))
    s[["teammate_id", "teammate_rank"]] = pd.DataFrame(tm_rows, index=s.index)

    # Main team, its car jump over the year before, engine and tyre supplier.
    main = entries.groupby(["driver_id", "year"]).agg(
        main_team=("team_id", _mode), engine_id=("engine_id", _mode), tyre_id=("tyre_id", _mode)
    )
    s = s.merge(main, left_on=["driver_id", "year"], right_index=True, how="left")
    team_car = (
        entries.assign(car=entries["team_season"].map(car_by_ts))
        .groupby(["team_id", "year"])["car"].mean()
    )
    s["car_jump"] = [
        team_car.get((t, y), np.nan) - team_car.get((t, y - 1), np.nan) for t, y in zip(s["main_team"], s["year"])
    ]

    # Rule changes this year or last.
    def rules_for(y):
        r = regs.loc[regs["year"].between(y - 1, y)]
        return "; ".join(f"{int(a)}: {b}" for a, b in zip(r["year"], r["change"]))

    s["rule_changes"] = s["year"].map(rules_for)
    return s.drop(columns=["package", "rival_package"])


def thresholds(factors: pd.DataFrame) -> dict[str, float]:
    """Cut-offs taken from title-winning seasons."""
    champs = factors.loc[(factors["champ_rank"] == 1) & (factors["starts"] >= MIN_STARTS)]
    return {
        "car_big": float(champs["car_gap"].quantile(2 / 3)),
        "rival_easy": float(champs["package_gap"].quantile(2 / 3)),
        "rival_tough": float(champs["package_gap"].quantile(1 / 3)),
    }


def summarise_window(rows: pd.DataFrame) -> dict:
    """Combine several seasons (e.g. a peak window) into one start-weighted factor set."""
    w = rows["starts"].clip(lower=1)
    out = {}
    for c in NUMERIC:
        v = rows[c].astype(float)
        ok = v.notna()
        out[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    out["starts"], out["wins"] = float(rows["starts"].sum()), float(rows["wins"].sum())
    out["car_jump"] = float(rows["car_jump"].iloc[0])  # the step into the first season
    out["n_seasons"] = len(rows)
    out["best_car_seasons"] = int((rows["car_rank"] == 1).sum())
    if rows["teammate_rank"].notna().any():
        best_tm = rows.loc[rows["teammate_rank"].idxmin()]
        out["teammate_rank"], out["teammate_name"] = best_tm["teammate_rank"], best_tm.get("teammate_name")
    else:
        out["teammate_rank"], out["teammate_name"] = np.nan, None
    toughest = rows.loc[rows["package_gap"].idxmin()] if rows["package_gap"].notna().any() else rows.iloc[0]
    for c in ("rival_id", "rival_name", "rival_team", "rival_same_car"):
        out[c] = toughest.get(c)
    for c in ("main_team", "team_name", "engine_id", "tyre_id"):
        out[c] = ", ".join(dict.fromkeys(rows[c].dropna().astype(str))) if c in rows else ""
    out["rule_changes"] = "; ".join(dict.fromkeys(x for r in rows["rule_changes"] for x in r.split("; ") if x))
    return out


def describe(f: dict, thr: dict[str, float]) -> dict:
    """Plain-language labels and details for each factor plus a one-line verdict."""
    cards = {}

    n, best = f.get("n_seasons", 1), f.get("best_car_seasons")
    rank = int(round(f["car_rank"])) if pd.notna(f["car_rank"]) else None
    if best is None:  # single season
        best = int(rank == 1)
    if best == n:
        car = "Best car by a big margin" if f["car_gap"] >= thr["car_big"] else "Best car"
    elif best > 0:
        car = f"Best car in {best} of {n} seasons"
    elif rank is not None and rank <= 3:
        car = "Top-3 car"
    else:
        car = "Not the best car"
    where = f"Car #{rank} on the grid" if n == 1 else f"Avg car rank #{f['car_rank']:.1f}"
    car_detail = f"{where} · {f['car_gap']:+.2f} vs the best other car"
    if pd.notna(f.get("teammate_rank")):
        car_detail += f" · teammate {f.get('teammate_name') or ''} finished P{int(f['teammate_rank'])}"
    cards["Car"] = (car, car_detail)

    srank = int(round(f["skill_rank"]))
    drv = "Best driver on the grid" if srank == 1 else ("Top-3 driver" if srank <= 3 else f"#{srank} driver on the grid")
    h2h = f"beat teammate in {f['h2h']:.0%} of battles" if pd.notna(f["h2h"]) else "no regular teammate"
    cards["Driver"] = (drv, f"Skill {f['driver_edge']:+.2f} vs average driver · {h2h}")

    rival = f.get("rival_name") or "–"
    rival_txt = f"{rival}{' (same team!)' if f.get('rival_same_car') else ''} · gap {f['package_gap']:+.2f}"
    if f["package_gap"] >= thr["rival_easy"]:
        comp = "Little competition"
    elif f["package_gap"] <= thr["rival_tough"]:
        comp = f"Tough rival: {rival}"
    else:
        comp = "Normal competition"
    cards["Competition"] = (comp, f"Closest rival (driver + car): {rival_txt}")

    diff = f["rivals_fail"] - f["own_fail"]
    if pd.isna(diff):
        rel = "Unknown"
    elif diff >= 0.08:
        rel = "Rivals broke down more"
    elif diff <= -0.08:
        rel = "Won despite failures" if f["wins"] > 0 else "Own car less reliable"
    else:
        rel = "Even reliability"
    cards["Reliability"] = (rel, f"Own car failures in {f['own_fail']:.0%} of races vs {f['rivals_fail']:.0%} for top rivals")

    jump = f["car_jump"]
    jump_txt = f"{f.get('team_name') or 'Team'} car {jump:+.2f} vs year before" if pd.notna(jump) else "no previous year"
    if f["rule_changes"]:
        rules = "New rules" + (" — team nailed them" if pd.notna(jump) and jump >= 0.05 else "")
        cards["New rules"] = (rules, f"{f['rule_changes']} · {jump_txt}")
    else:
        big_step = pd.notna(jump) and jump >= 0.10
        cards["New rules"] = ("Team made a big step" if big_step else "Stable rules", jump_txt)

    share = f["driver_share"]
    if pd.isna(share):
        verdict = "Neither driver nor car was above average"
    elif share >= 0.6:
        verdict = "Mostly the driver"
    elif share <= 0.4:
        verdict = "Mostly the car"
    else:
        verdict = "Driver and car together"
    extras = []
    if comp == "Little competition":
        extras.append("weak rivals")
    elif comp.startswith("Tough") and f["wins"] > 0:
        extras.append(f"beat a tough rival ({rival})")
    if rel == "Rivals broke down more":
        extras.append("rivals' cars broke down more")
    if cards["New rules"][0].endswith("nailed them"):
        extras.append("team got the new rules right")
    if extras:
        verdict += ", " + ", ".join(extras)
    return {"verdict": verdict, "driver_share": share, "cards": cards}

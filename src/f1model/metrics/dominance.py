"""Season-level dominance stats and the Peak Dominance Index.

Per driver-season:
- win / pole / podium rate
- points share: modern-rescored points as a share of the max (25 per start)
- margin over the field: rescored points lead over the next-best driver,
  as a share of the season's max (negative for everyone but the leader)
- teammate head-to-head in qualifying and races, plus average gap
- driver skill rating from the driver-vs-car model

The Dominance Index z-scores each component within its decade (so a driver is
measured against their own era) and takes a weighted sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_WEIGHTS = {"rating": 0.4, "h2h": 0.2, "points_share": 0.2, "margin": 0.2}
MIN_STARTS = 5
MIN_H2H = 6


def teammate_pairs(entries: pd.DataFrame) -> pd.DataFrame:
    """Every ordered (driver, teammate) pair in the same car at the same race."""
    cols = ["race_id", "year", "team_season", "driver_id", "quali_rank", "quali_pct", "finish_pct"]
    e = entries[cols]
    pairs = e.merge(e, on=["race_id", "year", "team_season"], suffixes=("", "_tm"))
    return pairs.loc[pairs["driver_id"] != pairs["driver_id_tm"]].reset_index(drop=True)


def teammate_h2h(entries: pd.DataFrame) -> pd.DataFrame:
    """Teammate battles won per driver-season (ties and non-comparable races skipped)."""
    p = teammate_pairs(entries)
    q_ok = p["quali_rank"].notna() & p["quali_rank_tm"].notna()
    r_ok = p["finish_pct"].notna() & p["finish_pct_tm"].notna() & (p["finish_pct"] != p["finish_pct_tm"])
    p["q_n"] = q_ok.astype(int)
    p["q_w"] = (q_ok & (p["quali_rank"] < p["quali_rank_tm"])).astype(int)
    p["r_n"] = r_ok.astype(int)
    p["r_w"] = (r_ok & (p["finish_pct"] > p["finish_pct_tm"])).astype(int)
    p["q_gap"] = (p["quali_pct"] - p["quali_pct_tm"]).where(q_ok)
    out = p.groupby(["driver_id", "year"]).agg(
        h2h_quali_w=("q_w", "sum"),
        h2h_quali_n=("q_n", "sum"),
        h2h_race_w=("r_w", "sum"),
        h2h_race_n=("r_n", "sum"),
        tm_quali_gap=("q_gap", "mean"),
        teammates=("driver_id_tm", "nunique"),
    )
    n = out["h2h_quali_n"] + out["h2h_race_n"]
    out["h2h"] = ((out["h2h_quali_w"] + out["h2h_race_w"]) / n).where(n >= MIN_H2H)
    return out.reset_index()


def season_stats(entries: pd.DataFrame) -> pd.DataFrame:
    """Results-based stats per driver-season."""
    e = entries.assign(
        win=entries["position"].eq(1),
        podium=entries["position"].le(3),
        pole=entries["pole"].fillna(False).astype(bool),
    )
    s = e.groupby(["driver_id", "year"]).agg(
        starts=("race_id", "size"),
        wins=("win", "sum"),
        podiums=("podium", "sum"),
        poles=("pole", "sum"),
        points=("points_modern", "sum"),
        teams=("team_id", lambda t: ", ".join(dict.fromkeys(t))),
    ).reset_index()
    for k in ("win", "podium", "pole"):
        s[f"{k}_rate"] = s[f"{k}s"] / s["starts"]
    s["points_share"] = s["points"] / (25 * s["starts"])

    season_races = entries.groupby("year")["race_id"].nunique()
    s["season_races"] = s["year"].map(season_races)
    s["champ_rank"] = s.groupby("year")["points"].rank(ascending=False, method="min").astype(int)
    top2 = s.groupby("year")["points"].apply(lambda x: x.nlargest(2).tolist())
    best = s["year"].map(top2.str[0])
    second = s["year"].map(lambda y: top2[y][1] if len(top2[y]) > 1 else 0)
    best_other = np.where(s["points"] >= best, second, best)
    s["margin"] = (s["points"] - best_other) / (25 * s["season_races"])
    return s


def _z_by_era(values: pd.Series, era: pd.Series) -> pd.Series:
    g = values.groupby(era)
    return (values - g.transform("mean")) / g.transform("std")


def dominance_table(entries: pd.DataFrame, driver_seasons: pd.DataFrame) -> pd.DataFrame:
    """Merge every season metric and compute the Dominance Index."""
    s = season_stats(entries)
    s = s.merge(teammate_h2h(entries), on=["driver_id", "year"], how="left")
    s = s.merge(driver_seasons[["driver_id", "year", "rating", "car", "career", "form"]], on=["driver_id", "year"])
    s["era"] = s["year"] // 10 * 10

    eligible = s["starts"] >= MIN_STARTS
    s["dominance"] = np.nan
    total = pd.Series(0.0, index=s.index[eligible])
    for col, w in INDEX_WEIGHTS.items():
        z = _z_by_era(s.loc[eligible, col], s.loc[eligible, "era"])
        s.loc[eligible, f"z_{col}"] = z
        total += w * z.fillna(0.0)  # missing h2h (too few battles) counts as average
    s.loc[eligible, "dominance"] = total
    return s.sort_values(["driver_id", "year"]).reset_index(drop=True)

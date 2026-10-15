"""Track-specific race prediction.

Track types
    street      F1DB marks the circuit as a street circuit
    high-speed  top third of permanent circuits by speed index
    mixed       middle third
    technical   bottom third
Speed index = pole lap speed / that season's median pole speed, median over
the circuit's races, so it is fair across eras. The labels are written to
data/circuits.csv; edit the `track_type` column there to override any of them.

Features (all known before the race starts)
    drv_form       driver's recent results (exp. weighted, 3-race half-life)
    drv_quali      driver's recent qualifying
    drv_vs_tm      driver's recent edge over their teammate (skill signal)
    team_form      car's recent results, both cars
    team_rel       car's recent mechanical-failure rate
    drv_err        driver's recent crash/spin rate
    type_adj       how much better/worse than usual the driver does on this
                   track type, shrunk toward 0 when there is little history
    circuit_adj    same for this exact circuit
    starts_log     experience
    quali_pct      grid slot (post-qualifying model only)

Model: linear (ridge for finishing order, logistic for win / podium). In the
2014-2026 backtest this matched or beat LightGBM, which overfit the noise, and
a 3-race form half-life beat 4 and 8 because cars change quickly in a season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRACK_TYPES = ["street", "high-speed", "mixed", "technical"]
HALFLIFE = 3
SHRINK = {"track_type": 6.0, "circuit_id": 3.0}

PRE_FEATURES = [
    "drv_form", "drv_quali", "drv_vs_tm", "team_form", "team_rel", "drv_err",
    "starts_log", "type_adj", "circuit_adj",
]
POST_FEATURES = PRE_FEATURES + ["quali_pct"]


# ---------------------------------------------------------------- track types
def build_circuit_types(races: pd.DataFrame, qualifying: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """Label every circuit street / high-speed / mixed / technical."""
    pole = qualifying.loc[qualifying["positionNumber"] == 1, ["raceId", "timeMillis"]]
    r = races.rename(columns={"id": "raceId"}).merge(pole, on="raceId", how="left")
    r["speed"] = r["courseLength"] / (r["timeMillis"] / 3_600_000)
    r["rel_speed"] = r["speed"] / r.groupby("year")["speed"].transform("median")
    idx = r.groupby("circuitId").agg(speed_index=("rel_speed", "median"), races_used=("rel_speed", "count"))

    c = circuits.rename(columns={"id": "circuit_id", "type": "f1db_type"})[["circuit_id", "name", "f1db_type"]]
    c = c.merge(idx, left_on="circuit_id", right_index=True, how="left")
    permanent = (c["f1db_type"] != "STREET") & c["speed_index"].notna()
    lo, hi = c.loc[permanent, "speed_index"].quantile([1 / 3, 2 / 3])
    c["track_type"] = np.select(
        [c["f1db_type"] == "STREET", c["speed_index"] >= hi, c["speed_index"] <= lo],
        ["street", "high-speed", "technical"],
        default="mixed",
    )
    return c.sort_values("circuit_id").reset_index(drop=True)


# ------------------------------------------------------------------ features
def _ewm_prior(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Exponentially weighted mean of each group's earlier rows (excludes the current row)."""
    return values.groupby(groups).transform(
        lambda s: s.ewm(halflife=HALFLIFE, ignore_na=True).mean().shift(1)
    )


def _shrunk_prior_mean(resid: pd.Series, keys: list[pd.Series], k: float) -> pd.Series:
    """Mean of earlier residuals in the same group, shrunk toward 0: sum / (n + k)."""
    has = resid.notna().astype(float)
    val = resid.fillna(0.0)
    csum = val.groupby(keys).cumsum() - val
    cnt = has.groupby(keys).cumsum() - has
    return csum / (cnt + k)


def build_features(entries: pd.DataFrame, track_types: pd.DataFrame) -> pd.DataFrame:
    """Pre-race features for every entry (rows must be in date order).

    Future races can be included as rows with NaN outcomes; their features are
    then the driver's state after the latest real race.
    """
    df = entries.sort_values(["date", "race_id", "finish_order"], na_position="last").reset_index(drop=True)
    tt = track_types.set_index("circuit_id")["track_type"]
    df["track_type"] = df["circuit_id"].map(tt).fillna("mixed")

    # Overall performance at a race: mean of race and qualifying percentile.
    df["perf"] = df[["finish_pct", "quali_pct"]].mean(axis=1)
    done = df["status"].notna()
    df["car_fail"] = (df["status"] == "car").astype(float).where(done)
    df["drv_fail"] = (df["status"] == "driver").astype(float).where(done)

    df["drv_form"] = _ewm_prior(df["perf"], df["driver_id"])
    df["drv_quali"] = _ewm_prior(df["quali_pct"], df["driver_id"])
    df["drv_err"] = _ewm_prior(df["drv_fail"], df["driver_id"])
    df["starts_log"] = np.log1p(df.groupby("driver_id").cumcount())

    # Edge over teammate(s) at each race, then smoothed over earlier races.
    team_race = df.groupby(["race_id", "team_id"])
    tm_sum = team_race["perf"].transform("sum")
    tm_cnt = team_race["perf"].transform("count")
    df["tm_edge"] = (df["perf"] - (tm_sum - df["perf"]) / (tm_cnt - 1)).where(tm_cnt > 1)
    df["drv_vs_tm"] = _ewm_prior(df["tm_edge"], df["driver_id"])

    # Car form and reliability: one value per (race, team), smoothed over the team's races.
    tr = df.groupby(["race_id", "team_id"], sort=False).agg(
        date=("date", "first"), t_perf=("perf", "mean"), t_fail=("car_fail", "mean")
    ).reset_index().sort_values("date", kind="stable")
    tr["team_form"] = _ewm_prior(tr["t_perf"], tr["team_id"])
    tr["team_rel"] = _ewm_prior(tr["t_fail"], tr["team_id"])
    df = df.merge(tr[["race_id", "team_id", "team_form", "team_rel"]], on=["race_id", "team_id"], how="left")

    # Track-type and circuit adjustments: earlier over/under-performance vs own form.
    resid = df["perf"] - df["drv_form"]
    for col, out in (("track_type", "type_adj"), ("circuit_id", "circuit_adj")):
        df[out] = _shrunk_prior_mean(resid, [df["driver_id"], df[col]], SHRINK[col])

    df["win"] = (df["position"] == 1).astype(float).where(done)
    df["podium"] = (df["position"] <= 3).astype(float).where(done)
    return df


# --------------------------------------------------------------------- model
class TrackModel:
    """Finishing-percentile regressor plus win and podium classifiers."""

    def __init__(self, features: list[str]):
        self.features = features

        def prep():
            # Rookies have no form yet: fill with the median, then standardise.
            return [SimpleImputer(strategy="median"), StandardScaler()]

        self.reg = make_pipeline(*prep(), Ridge(alpha=1.0))
        self.win = make_pipeline(*prep(), LogisticRegression(max_iter=1000))
        self.pod = make_pipeline(*prep(), LogisticRegression(max_iter=1000))

    def fit(self, df: pd.DataFrame) -> "TrackModel":
        train = df.loc[df["race_pct"].notna()]
        X = train[self.features]
        self.reg.fit(X, train["race_pct"])
        self.win.fit(X, train["win"].astype(int))
        self.pod.fit(X, train["podium"].astype(int))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        out = df[["race_id", "driver_id", "team_id"]].copy()
        out["score"] = self.reg.predict(X)
        out["p_win"] = self.win.predict_proba(X)[:, 1]
        out["p_podium"] = self.pod.predict_proba(X)[:, 1]
        # Make probabilities consistent within each race: wins sum to 1, podiums to 3.
        g = out.groupby("race_id")
        out["p_win"] = out["p_win"] / g["p_win"].transform("sum")
        out["p_podium"] = (3 * out["p_podium"] / g["p_podium"].transform("sum")).clip(upper=1.0)
        out["pred_rank"] = g["score"].rank(ascending=False, method="first").astype(int)
        return out


# ------------------------------------------------------------ upcoming races
def upcoming_entries(entries: pd.DataFrame, races: pd.DataFrame, n_races: int | None = None) -> pd.DataFrame:
    """Placeholder rows for races with no results yet, using the latest grid."""
    done = set(entries["race_id"])
    future = races.loc[~races["id"].isin(done) & (races["year"] >= entries["year"].max())].sort_values("date")
    if n_races:
        future = future.head(n_races)
    last = entries.loc[entries["race_id"] == entries.loc[entries["date"].idxmax(), "race_id"]]
    rows = []
    for _, race in future.iterrows():
        for _, e in last.iterrows():
            rows.append(
                {
                    "race_id": race["id"], "year": race["year"], "round": race["round"],
                    "date": race["date"], "circuit_id": race["circuitId"],
                    "driver_id": e["driver_id"], "team_id": e["team_id"],
                }
            )
    return pd.DataFrame(rows)

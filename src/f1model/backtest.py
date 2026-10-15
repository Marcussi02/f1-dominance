"""Walk-forward backtest: train on every season before Y, predict season Y race by race.

Compared against two simple baselines:
    grid order   start positions (what the post-qualifying model must beat)
    form order   driver form + car form, no model (for the pre-qualifying model)
"""

from __future__ import annotations

import pandas as pd

from f1model.models.track import POST_FEATURES, PRE_FEATURES, TrackModel

TRAIN_FROM = 1980


def _race_metrics(df: pd.DataFrame, score: pd.Series, model: str) -> pd.DataFrame:
    d = df.assign(score=score.to_numpy())
    d["pred_rank"] = d.groupby("race_id")["score"].rank(ascending=False, method="first")
    rows = []
    for race_id, g in d.groupby("race_id"):
        rows.append(
            {
                "race_id": race_id,
                "year": g["year"].iat[0],
                "model": model,
                "spearman": g["pred_rank"].corr(g["finish_order"], method="spearman"),
                "mae": (g["pred_rank"] - g["finish_order"]).abs().mean(),
                "winner_hit": float(g.loc[g["pred_rank"] == 1, "position"].eq(1).any()),
            }
        )
    return pd.DataFrame(rows)


def backtest(features: pd.DataFrame, test_years: list[int]) -> pd.DataFrame:
    """Per-race accuracy for the models and baselines across the test seasons."""
    data = features.loc[features["race_pct"].notna() & (features["year"] >= TRAIN_FROM)]
    out = []
    for year in test_years:
        train = data.loc[data["year"] < year]
        test = data.loc[data["year"] == year]
        if test.empty:
            continue
        pre = TrackModel(PRE_FEATURES).fit(train).predict(test)
        post = TrackModel(POST_FEATURES).fit(train).predict(test)
        out += [
            _race_metrics(test, pre["score"], "model (before qualifying)"),
            _race_metrics(test, post["score"], "model (after qualifying)"),
            _race_metrics(test, -test["quali_rank"].fillna(99), "baseline: grid order"),
            _race_metrics(test, test["drv_form"].fillna(0.3) + test["team_form"].fillna(0.3), "baseline: recent form"),
        ]
    return pd.concat(out, ignore_index=True)


def summarise(per_race: pd.DataFrame) -> pd.DataFrame:
    """Average accuracy per model over all test races."""
    return (
        per_race.groupby("model")
        .agg(races=("race_id", "nunique"), spearman=("spearman", "mean"), mae=("mae", "mean"), winner_hit=("winner_hit", "mean"))
        .sort_values("spearman", ascending=False)
        .round(3)
    )

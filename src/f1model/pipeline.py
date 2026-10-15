"""Run the full model and write everything the dashboard needs to data/out/.

    python -m f1model.pipeline            # use cached data
    python -m f1model.pipeline --refresh  # re-download F1DB first
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from f1model.backtest import backtest, summarise
from f1model.data.clean import build_entries
from f1model.data.fetch import DATA_DIR, download, load_table
from f1model.metrics.dominance import dominance_table
from f1model.models.rating import fit_ratings, peak_window
from f1model.models.track import (
    PRE_FEATURES, TRACK_TYPES, TrackModel, build_circuit_types, build_features, upcoming_entries,
)

OUT_DIR = DATA_DIR / "out"
CIRCUITS_CSV = DATA_DIR / "circuits.csv"
BACKTEST_YEARS = list(range(2014, 2027))


def circuit_types(races: pd.DataFrame) -> pd.DataFrame:
    """Load data/circuits.csv (keeping manual edits) or build it on first run."""
    if CIRCUITS_CSV.exists():
        return pd.read_csv(CIRCUITS_CSV)
    c = build_circuit_types(races, load_table("qualifying"), load_table("circuits"))
    c.round({"speed_index": 4}).to_csv(CIRCUITS_CSV, index=False)
    return c


def predict_tracks(entries: pd.DataFrame, races: pd.DataFrame, track_types: pd.DataFrame):
    """Predict the rest of the calendar, plus every recently used circuit, for the current grid.

    Returns (predictions, per-driver track-type strengths, historical feature frame).
    """
    upcoming = upcoming_entries(entries, races)
    upcoming["upcoming"] = True
    last_race = entries.loc[entries["date"] == entries["date"].max()]
    grid = last_race[["driver_id", "team_id"]].drop_duplicates("driver_id")

    # "What if the next race were here?" for circuits not left on this year's calendar.
    recent = entries.loc[entries["year"] >= entries["year"].max() - 6, "circuit_id"].unique()
    extra = sorted(set(recent) - set(upcoming.get("circuit_id", [])))
    next_day = (pd.to_datetime(entries["date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hypo = pd.DataFrame(
        [
            {"race_id": -(i + 1), "year": entries["year"].max(), "round": 0, "date": next_day,
             "circuit_id": c, "driver_id": d, "team_id": t, "upcoming": False}
            for i, c in enumerate(extra) for d, t in grid.itertuples(index=False)
        ]
    )
    future = pd.concat([upcoming, hypo], ignore_index=True)

    feats = build_features(pd.concat([entries, future], ignore_index=True), track_types)
    is_future = feats["race_id"].isin(future["race_id"])
    # Placeholder rows must not count as experience for each other.
    feats.loc[is_future, "starts_log"] = feats.loc[is_future].groupby("driver_id")["starts_log"].transform("min")

    model = TrackModel(PRE_FEATURES).fit(feats.loc[~is_future & (feats["year"] >= 1980)])
    target = feats.loc[is_future].astype({"upcoming": bool})
    keep = ["race_id", "driver_id", "circuit_id", "track_type", "date", "upcoming", "type_adj", "circuit_adj", "drv_form", "team_form"]
    pred = model.predict(target).merge(target[keep], on=["race_id", "driver_id"])

    # Each current driver's strength on each track type (as of the latest race).
    strengths = pd.concat(
        [
            pred.loc[pred["track_type"] == tt].drop_duplicates("driver_id")[["driver_id", "type_adj"]].assign(track_type=tt)
            for tt in TRACK_TYPES
        ],
        ignore_index=True,
    )
    return pred, strengths, feats.loc[~is_future]


def summarise_peaks(seasons: pd.DataFrame) -> pd.DataFrame:
    """Best 3-season window per driver for the Dominance Index and the pure skill rating."""
    usable = seasons.dropna(subset=["dominance"])
    peaks = peak_window(usable, "dominance")
    skill = peak_window(usable, "rating").rename(
        columns={"peak_score": "peak_rating", "peak_start": "rating_start", "peak_end": "rating_end"}
    )[["driver_id", "peak_rating", "rating_start", "rating_end"]]
    peaks = peaks.merge(skill, on="driver_id")

    in_peak = seasons.merge(peaks[["driver_id", "peak_start", "peak_end"]], on="driver_id")
    in_peak = in_peak.loc[in_peak["year"].between(in_peak["peak_start"], in_peak["peak_end"])]
    stats = in_peak.groupby("driver_id").agg(
        peak_wins=("wins", "sum"),
        peak_poles=("poles", "sum"),
        peak_titles=("champ_rank", lambda r: int((r == 1).sum())),
        peak_h2h=("h2h", "mean"),
        peak_car=("car", "mean"),
        peak_teams=("teams", lambda t: ", ".join(dict.fromkeys(", ".join(t).split(", ")))),
        peak_teammates=("teammates", "sum"),
    )
    return peaks.merge(stats, on="driver_id").sort_values("peak_score", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download F1DB")
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()
    run(refresh=args.refresh, with_backtest=not args.skip_backtest)


def run(refresh: bool = False, with_backtest: bool = True) -> None:
    """Build every output file. Also called by the dashboard on a fresh server."""
    if refresh:
        download(force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    races = load_table("races")
    entries = build_entries(load_table("results"), races, load_table("entrants"))
    names = load_table("drivers").set_index("id")["name"]
    teams = load_table("constructors").set_index("id")["name"]
    print(f"entries: {len(entries):,} starts, {entries['year'].min()}-{entries['year'].max()}")

    # 1. Dominance
    ratings = fit_ratings(entries)
    seasons = dominance_table(entries, ratings["driver_seasons"])
    seasons["driver"] = seasons["driver_id"].map(names)
    peaks = summarise_peaks(seasons)
    peaks["driver"] = peaks["driver_id"].map(names)
    seasons.to_parquet(OUT_DIR / "seasons.parquet", index=False)
    peaks.to_parquet(OUT_DIR / "peaks.parquet", index=False)
    print("\nTop 10 peaks:")
    print(peaks.head(10)[["driver", "peak_start", "peak_end", "peak_score", "peak_rating", "peak_wins"]].round(3).to_string(index=False))

    # 2. Track prediction
    track_types = circuit_types(races)
    pred, strengths, feats = predict_tracks(entries, races, track_types)
    pred["race_name"] = pred["race_id"].map(races.set_index("id")["officialName"])
    pred["circuit"] = pred["circuit_id"].map(track_types.set_index("circuit_id")["name"])
    pred["driver"] = pred["driver_id"].map(names)
    pred["team"] = pred["team_id"].map(teams)
    strengths["driver"] = strengths["driver_id"].map(names)
    pred.to_parquet(OUT_DIR / "predictions.parquet", index=False)
    strengths.to_parquet(OUT_DIR / "track_strengths.parquet", index=False)

    nxt = pred.loc[pred["upcoming"]].sort_values(["date", "pred_rank"])
    if len(nxt):
        first = nxt.loc[nxt["race_id"] == nxt["race_id"].iloc[0]]
        print(f"\nNext race: {first['race_name'].iat[0]} ({first['track_type'].iat[0]})")
        print(first.head(5)[["pred_rank", "driver", "team", "p_win", "p_podium"]].round(3).to_string(index=False))

    # 3. Backtest
    meta = {"last_race": str(entries["date"].max()), "starts": len(entries), "backtest_years": BACKTEST_YEARS}
    if with_backtest:
        per_race = backtest(feats, BACKTEST_YEARS)
        per_race.to_parquet(OUT_DIR / "backtest.parquet", index=False)
        summary = summarise(per_race)
        print(f"\nBacktest {BACKTEST_YEARS[0]}-{BACKTEST_YEARS[-1]}:")
        print(summary.to_string())
        meta["backtest"] = summary.reset_index().to_dict(orient="records")
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()

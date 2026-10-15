"""Turn raw F1DB results into one clean row per race start.

Key choices
- Only real starters count (finished, DNF, NC or DSQ). DNQ/DNS/etc are dropped.
- The Indianapolis 500 (in the championship 1950-60) is dropped: a separate
  series with its own drivers.
- Shared drives (1950s car swaps) are dropped.
- Every retirement is labelled: driver error, car failure, racing incident or
  other. Only driver errors count against a driver.
- Points are re-scored with the modern 25-18-15... system so eras compare.
- Results are turned into field-size-independent percentiles (1 = best).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODERN_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

STARTER_CODES = {"DNF", "NC", "DSQ"}

DRIVER_ERRORS = {
    "accident",
    "spun off",
    "spin",
    "stalled",
    "accident damage",
    "accident on formation lap",
    "driving too slowly",
    "rejoined track illegally",
    "overtaking on formation lap",
}
INCIDENTS = {"collision", "collision damage", "puncture", "puncture damage", "fatal collision"}
# Retirements that say nothing about the driver's pace or the car's reliability.
OTHER_PREFIXES = (
    "withdrew", "physical", "injury", "unwell", "illness", "unfit", "fatal",
    "car raced by", "illegal", "ignored", "technical infringements", "money dispute",
    "car underweight", "missed weight check", "received outside assistance",
    "push start", "failed", "misled", "practice only", "reserve driver",
    "safety concerns", "tyre dispute", "reversed in pits", "incorrect",
    "caused collision", "accident in", "injury in", "car damaged in", "camera car",
    "car seized", "car not ready", "car unraceworthy", "retirement", "tyre shortage",
    "additional brake fluid", "fuel sample", "wrong tyres",
)


def classify_retirement(position_text: str, reason: object) -> str:
    """Label a result as finished / driver / car / incident / other."""
    if position_text == "DSQ":
        return "other"
    if position_text != "DNF":
        return "finished"
    if not isinstance(reason, str) or not reason.strip():
        return "other"
    r = reason.strip().lower()
    if r in DRIVER_ERRORS:
        return "driver"
    if r in INCIDENTS:
        return "incident"
    if r.startswith(OTHER_PREFIXES):
        return "other"
    return "car"


def modern_points(position: float) -> int:
    """Points under the 2010+ system, ignoring fastest-lap and sprint bonuses."""
    if pd.isna(position):
        return 0
    return MODERN_POINTS.get(int(position), 0)


def percentile(rank: pd.Series, field: pd.Series) -> pd.Series:
    """Map a 1-based rank in a field of n to [0, 1], with 1 for the winner."""
    return (1 - (rank - 1) / (field - 1).clip(lower=1)).astype(float)


def entrant_by_round(entrants: pd.DataFrame) -> pd.DataFrame:
    """Expand F1DB's season entry list to one row per (year, round, constructor, driver)."""
    e = entrants.loc[~entrants["testDriver"].fillna(False).astype(bool)].copy()
    e["round"] = e["rounds"].astype(str).str.split(";")
    e = e.explode("round")
    e = e.loc[e["round"].str.isdigit()]
    e["round"] = e["round"].astype(int)
    keys = ["year", "round", "constructorId", "driverId"]
    return e.drop_duplicates(keys)[keys + ["entrantId"]]


def build_entries(results: pd.DataFrame, races: pd.DataFrame, entrants: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per race start with clean, comparable outcome columns.

    With `entrants`, a "car" is entrant + constructor, so a privateer running a
    customer Lotus is not treated as a teammate of the works Team Lotus.
    """
    race_info = races.rename(columns={"id": "raceId"})[
        ["raceId", "date", "grandPrixId", "circuitId", "circuitType", "courseLength"]
    ]
    df = results.merge(race_info, on="raceId", how="left")
    if entrants is not None:
        df = df.merge(entrant_by_round(entrants), on=["year", "round", "constructorId", "driverId"], how="left")
    else:
        df["entrantId"] = np.nan
    df["entrantId"] = df["entrantId"].fillna(df["constructorId"])

    is_starter = df["positionText"].astype(str).str.isdigit() | df["positionText"].isin(STARTER_CODES)
    keep = is_starter & (df["grandPrixId"] != "indianapolis") & ~df["sharedCar"].fillna(False).astype(bool)
    df = df.loc[keep].copy()

    df["status"] = [classify_retirement(p, r) for p, r in zip(df["positionText"], df["reasonRetired"])]
    df["classified"] = df["positionNumber"].notna()
    df["points_modern"] = df["positionNumber"].map(modern_points)

    g = df.groupby("raceId")
    df["n_starters"] = g["driverId"].transform("size")
    df["finish_order"] = g["positionDisplayOrder"].rank(method="first")
    df["race_pct"] = percentile(df["finish_order"], df["n_starters"])

    q = df["qualificationPositionNumber"].fillna(df["gridPositionNumber"])
    df["quali_rank"] = q.groupby(df["raceId"]).rank(method="first")
    df["n_qualified"] = q.notna().groupby(df["raceId"]).transform("sum")
    df["quali_pct"] = percentile(df["quali_rank"], df["n_qualified"])
    df.loc[q.isna(), "quali_pct"] = np.nan

    # Race pace measure for ratings: driver errors count as last place,
    # car failures / incidents / other say nothing about the driver.
    df["finish_pct"] = df["race_pct"].where(df["status"] == "finished")
    df.loc[df["status"] == "driver", "finish_pct"] = 0.0

    df["team_season"] = df["entrantId"] + "|" + df["constructorId"] + "_" + df["year"].astype(str)
    df["driver_season"] = df["driverId"] + "_" + df["year"].astype(str)

    out = df.rename(
        columns={
            "raceId": "race_id",
            "driverId": "driver_id",
            "constructorId": "team_id",
            "entrantId": "entrant_id",
            "circuitId": "circuit_id",
            "circuitType": "f1db_circuit_type",
            "positionNumber": "position",
            "gridPositionNumber": "grid",
            "polePosition": "pole",
        }
    )
    cols = [
        "race_id", "year", "round", "date", "circuit_id", "f1db_circuit_type", "courseLength",
        "driver_id", "team_id", "entrant_id", "team_season", "driver_season", "position", "grid", "pole",
        "status", "classified", "points_modern", "n_starters", "finish_order", "race_pct",
        "finish_pct", "quali_rank", "quali_pct", "laps",
    ]
    return out[cols].sort_values(["date", "race_id", "finish_order"]).reset_index(drop=True)

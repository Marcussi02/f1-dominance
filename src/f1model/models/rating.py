"""Driver-vs-car rating model.

Every race start gives up to two observations: qualifying percentile and race
percentile (1 = best in the field). Each observation is modelled as

    result = driver career skill + driver's form that season + car strength that season

and fitted with ridge regression. Teammates share the car term, so the gap
between them pins down driver skill; drivers who move teams link the whole
grid together across seasons and eras.

Different shrinkage per term (via column scaling):
- car (team-season): weak shrinkage, cars differ a lot
- driver career:     moderate shrinkage
- season form:       strong shrinkage, so one odd season can't dominate
Ratings are in percentile units: +0.10 means finishing 10% of the field
higher than an average driver in the same car.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

PENALTY = {"driver": 3.0, "season_form": 6.0, "car": 1.5}


def _observations(entries: pd.DataFrame) -> pd.DataFrame:
    base = ["race_id", "year", "driver_id", "driver_season", "team_season"]
    race = entries.loc[entries["finish_pct"].notna(), base + ["finish_pct"]].rename(columns={"finish_pct": "y"})
    quali = entries.loc[entries["quali_pct"].notna(), base + ["quali_pct"]].rename(columns={"quali_pct": "y"})
    race["is_quali"] = 0.0
    quali["is_quali"] = 1.0
    return pd.concat([race, quali], ignore_index=True)


def _one_hot(codes: np.ndarray, n_cols: int, scale: float) -> sparse.csr_matrix:
    rows = np.arange(len(codes))
    return sparse.csr_matrix((np.full(len(codes), scale), (rows, codes)), shape=(len(codes), n_cols))


def fit_ratings(entries: pd.DataFrame, penalty: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    """Fit the model; return driver-season, driver-career and car ratings."""
    pen = {**PENALTY, **(penalty or {})}
    obs = _observations(entries)

    blocks, keys = [], {}
    for name, col in (("driver", "driver_id"), ("season_form", "driver_season"), ("car", "team_season")):
        codes, uniques = pd.factorize(obs[col])
        keys[name] = uniques
        # Scaling a column by s turns a penalty of 1 into a penalty of 1/s^2.
        blocks.append(_one_hot(codes, len(uniques), 1.0 / np.sqrt(pen[name])))
    X = sparse.hstack(blocks + [sparse.csr_matrix(obs[["is_quali"]].to_numpy())]).tocsr()

    model = Ridge(alpha=1.0, fit_intercept=True, solver="sparse_cg", max_iter=5000, tol=1e-6)
    model.fit(X, obs["y"].to_numpy())

    coefs, start = {}, 0
    for name in ("driver", "season_form", "car"):
        n = len(keys[name])
        coefs[name] = pd.Series(model.coef_[start : start + n] / np.sqrt(pen[name]), index=keys[name])
        start += n

    car = coefs["car"].rename("car").rename_axis("team_season").reset_index()
    career = coefs["driver"].rename("career").rename_axis("driver_id").reset_index()

    # Driver-season table: skill = career + season form; car = race-weighted car rating.
    ds = entries.groupby(["driver_id", "year", "driver_season"]).size().rename("starts").reset_index()
    ds["career"] = ds["driver_id"].map(coefs["driver"])
    ds["form"] = ds["driver_season"].map(coefs["season_form"]).fillna(0.0)
    ds["rating"] = ds["career"] + ds["form"]
    team_mix = entries.groupby(["driver_season", "team_season"]).size().rename("n").reset_index()
    team_mix["car"] = team_mix["team_season"].map(coefs["car"])
    car_by_ds = team_mix.groupby("driver_season").apply(
        lambda t: np.average(t["car"], weights=t["n"]), include_groups=False
    )
    ds["car"] = ds["driver_season"].map(car_by_ds)
    n_obs = obs.groupby("driver_season").size()
    ds["n_obs"] = ds["driver_season"].map(n_obs).fillna(0).astype(int)

    return {
        "driver_seasons": ds.drop(columns="driver_season"),
        "careers": career,
        "cars": car,
        "intercept": pd.DataFrame({"intercept": [model.intercept_], "quali_shift": [model.coef_[-1]]}),
    }


def peak_window(seasons: pd.DataFrame, score: str, window: int = 3, min_starts: int = 5) -> pd.DataFrame:
    """Best run of `window` consecutive raced seasons per driver, start-weighted.

    Seasons with fewer than `min_starts` starts are skipped. Drivers with fewer
    usable seasons than `window` get their full-career average and are flagged.
    """
    usable = seasons.loc[seasons["starts"] >= min_starts].sort_values(["driver_id", "year"])
    rows = []
    for driver, g in usable.groupby("driver_id", sort=False):
        vals = g[score].to_numpy()
        w = g["starts"].to_numpy()
        years = g["year"].to_numpy()
        k = min(window, len(g))
        best, best_i = -np.inf, 0
        for i in range(len(g) - k + 1):
            m = np.average(vals[i : i + k], weights=w[i : i + k])
            if m > best:
                best, best_i = m, i
        rows.append(
            {
                "driver_id": driver,
                "peak_score": best,
                "peak_start": int(years[best_i]),
                "peak_end": int(years[best_i + k - 1]),
                "peak_starts": int(w[best_i : best_i + k].sum()),
                "short_career": len(g) < window,
            }
        )
    return pd.DataFrame(rows)

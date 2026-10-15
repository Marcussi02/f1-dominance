import numpy as np
import pandas as pd
import pytest

from f1model.data.clean import build_entries, classify_retirement, modern_points, percentile
from f1model.metrics.dominance import season_stats, teammate_h2h
from f1model.models.rating import fit_ratings, peak_window
from f1model.models.track import _ewm_prior, _shrunk_prior_mean, build_circuit_types


# ------------------------------------------------------------------ cleaning
@pytest.mark.parametrize(
    "text, reason, expected",
    [
        ("1", None, "finished"),
        ("NC", None, "finished"),
        ("DNF", "Engine", "car"),
        ("DNF", "Spun off", "driver"),
        ("DNF", "Collision", "incident"),
        ("DNF", "Withdrew", "other"),
        ("DNF", "Car raced by Hobbs", "other"),
        ("DSQ", None, "other"),
    ],
)
def test_classify_retirement(text, reason, expected):
    assert classify_retirement(text, reason) == expected


def test_modern_points_rescores_any_era():
    assert [modern_points(p) for p in (1, 2, 10, 11, np.nan)] == [25, 18, 1, 0, 0]


def test_percentile_maps_winner_to_one_and_last_to_zero():
    out = percentile(pd.Series([1, 5, 9]), pd.Series([9, 9, 9]))
    assert out.tolist() == [1.0, 0.5, 0.0]


def _raw(rows):
    cols = ["raceId", "year", "round", "positionDisplayOrder", "positionNumber", "positionText", "driverId",
            "constructorId", "sharedCar", "reasonRetired", "points", "polePosition",
            "qualificationPositionNumber", "gridPositionNumber", "laps"]
    return pd.DataFrame(rows, columns=cols)


def test_build_entries_filters_and_scores():
    races = pd.DataFrame(
        {"id": [1, 2], "year": [1955, 1955], "date": ["1955-05-01", "1955-05-30"],
         "grandPrixId": ["monaco", "indianapolis"], "circuitId": ["monaco", "indianapolis"],
         "circuitType": ["STREET", "RACE"], "courseLength": [3.1, 4.0]}
    )
    results = _raw([
        (1, 1955, 1, 1, 1, "1", "a", "t1", False, None, 8, True, 1, 1, 100),
        (1, 1955, 1, 2, 2, "2", "b", "t1", False, None, 6, False, 2, 2, 100),
        (1, 1955, 1, 3, None, "DNF", "c", "t2", False, "Accident", 0, False, 3, 3, 10),
        (1, 1955, 1, 4, None, "DNF", "d", "t2", False, "Engine", 0, False, 4, 4, 5),
        (1, 1955, 1, 5, None, "DNQ", "e", "t3", False, None, 0, False, None, None, 0),
        (2, 1955, 2, 1, 1, "1", "indy", "kurtis", False, None, 8, True, 1, 1, 200),
    ])
    e = build_entries(results, races)
    assert set(e["driver_id"]) == {"a", "b", "c", "d"}  # DNQ and Indy 500 dropped
    by = e.set_index("driver_id")
    assert by.loc["a", "points_modern"] == 25 and by.loc["b", "points_modern"] == 18
    assert by.loc["c", "finish_pct"] == 0.0  # driver error counts as last
    assert np.isnan(by.loc["d", "finish_pct"])  # engine failure not held against driver
    assert by.loc["a", "race_pct"] == 1.0 and by.loc["d", "race_pct"] == 0.0


# ------------------------------------------------------------------- metrics
def _entries(rows):
    cols = ["race_id", "year", "driver_id", "team_id", "team_season", "position", "pole",
            "quali_rank", "quali_pct", "finish_pct", "points_modern"]
    return pd.DataFrame(rows, columns=cols)


def test_teammate_h2h_counts_only_comparable_battles():
    e = _entries([
        # race 1: a beats b in both
        (1, 2000, "a", "t", "t_2000", 1, True, 1, 1.0, 1.0, 25),
        (1, 2000, "b", "t", "t_2000", 2, False, 2, 0.8, 0.8, 18),
        # race 2: b out-qualifies a; a's car failed so the race is not comparable
        (2, 2000, "a", "t", "t_2000", None, False, 2, 0.8, np.nan, 0),
        (2, 2000, "b", "t", "t_2000", 1, True, 1, 1.0, 1.0, 25),
    ])
    h = teammate_h2h(e).set_index("driver_id")
    assert (h.loc["a", "h2h_quali_w"], h.loc["a", "h2h_quali_n"]) == (1, 2)
    assert (h.loc["a", "h2h_race_w"], h.loc["a", "h2h_race_n"]) == (1, 1)
    assert (h.loc["b", "h2h_race_w"], h.loc["b", "h2h_race_n"]) == (0, 1)


def test_season_margin_is_lead_over_next_driver():
    e = _entries([
        (1, 2000, "a", "t", "t_2000", 1, True, 1, 1.0, 1.0, 25),
        (1, 2000, "b", "u", "u_2000", 2, False, 2, 0.5, 0.5, 18),
        (2, 2000, "a", "t", "t_2000", 1, True, 1, 1.0, 1.0, 25),
        (2, 2000, "b", "u", "u_2000", 2, False, 2, 0.5, 0.5, 18),
    ])
    s = season_stats(e).set_index("driver_id")
    assert s.loc["a", "margin"] == pytest.approx((50 - 36) / 50)
    assert s.loc["b", "margin"] == pytest.approx((36 - 50) / 50)
    assert s.loc["a", "points_share"] == 1.0


# -------------------------------------------------------------------- rating
def test_rating_separates_driver_from_car():
    """Synthetic grid: drivers swap teams each season; skill must be recovered, not car."""
    rng = np.random.default_rng(0)
    skill = {f"d{i}": s for i, s in enumerate([0.3, 0.2, 0.1, 0.0, -0.1, -0.2])}
    rows, race = [], 0
    for year in range(2000, 2006):
        car = dict(zip("ABC", rng.permutation([0.3, 0.0, -0.3])))
        drivers = rng.permutation(list(skill))
        for _ in range(15):
            race += 1
            for k, d in enumerate(drivers):
                team = "ABC"[k // 2]
                y = 0.5 + skill[d] + car[team] + rng.normal(0, 0.05, 2)
                rows.append({"race_id": race, "year": year, "driver_id": d, "team_season": f"{team}_{year}",
                             "driver_season": f"{d}_{year}", "finish_pct": y[0], "quali_pct": y[1]})
    r = fit_ratings(pd.DataFrame(rows))
    career = r["careers"].set_index("driver_id")["career"]
    assert career.rank().corr(pd.Series(skill).rank(), method="spearman") > 0.95


def test_peak_window_picks_best_consecutive_seasons():
    s = pd.DataFrame({
        "driver_id": ["x"] * 5,
        "year": [2001, 2002, 2003, 2004, 2005],
        "starts": [10, 10, 10, 10, 2],  # 2005 has too few starts and is skipped
        "score": [0.0, 1.0, 2.0, 3.0, 9.0],
    })
    p = peak_window(s, "score").iloc[0]
    assert (p["peak_start"], p["peak_end"]) == (2002, 2004)
    assert p["peak_score"] == pytest.approx(2.0)
    assert not p["short_career"]


# --------------------------------------------------------------------- track
def test_ewm_prior_never_uses_current_race():
    v = pd.Series([1.0, 0.0, 0.0])
    out = _ewm_prior(v, pd.Series(["a", "a", "a"]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1.0


def test_shrunk_prior_mean():
    resid = pd.Series([0.2, 0.2, np.nan, 0.2])
    out = _shrunk_prior_mean(resid, [pd.Series(["a"] * 4)], k=2.0)
    assert out.tolist() == pytest.approx([0.0, 0.2 / 3, 0.4 / 4, 0.4 / 4])


def test_circuit_types():
    circuits = pd.DataFrame({"id": ["s", "fast", "mid", "slow"], "name": list("SFMT"), "type": ["STREET", "RACE", "RACE", "RACE"]})
    races = pd.DataFrame({"id": [1, 2, 3, 4], "year": [2000] * 4, "circuitId": ["s", "fast", "mid", "slow"], "courseLength": [3.0, 6.0, 4.0, 3.0]})
    quali = pd.DataFrame({"raceId": [1, 2, 3, 4], "positionNumber": [1] * 4, "timeMillis": [90_000] * 4})
    t = build_circuit_types(races, quali, circuits).set_index("circuit_id")["track_type"]
    assert t.to_dict() == {"fast": "high-speed", "mid": "mixed", "s": "street", "slow": "technical"}

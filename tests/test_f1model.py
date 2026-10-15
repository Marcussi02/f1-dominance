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


# ------------------------------------------------------------------- explain
from f1model.metrics.explain import describe, driver_share, season_factors, summarise_window  # noqa: E402

THR = {"car_big": 0.05, "rival_easy": 0.04, "rival_tough": -0.01}


def test_driver_share_ignores_below_average_parts():
    out = driver_share(pd.Series([0.2, 0.1, -0.1, -0.1]), pd.Series([0.2, -0.3, 0.3, -0.2]))
    assert out.iloc[0] == 0.5 and out.iloc[1] == 1.0 and out.iloc[2] == 0.0 and np.isnan(out.iloc[3])


def _explain_inputs():
    """2 seasons, 3 cars: A (best), B, C. Driver a in A, rival b in B, c and d share C."""
    rows = []
    for year in (2013, 2014):
        for race in range(6):
            for d, team in (("a", "A"), ("b", "B"), ("c", "C"), ("d", "C")):
                fail = d == "b" and race < 3  # b's car breaks down half the time
                rows.append({"race_id": year * 10 + race, "year": year, "driver_id": d, "team_id": team,
                             "team_season": f"{team}_{year}", "engine_id": "eng", "tyre_id": "tyr",
                             "status": "car" if fail else "finished"})
    entries = pd.DataFrame(rows)
    car = {"A": 0.30, "B": 0.10, "C": 0.0}
    cars = pd.DataFrame([{"team_season": f"{t}_{y}", "car": v + (0.1 if (t == "A" and y == 2014) else 0)}
                         for y in (2013, 2014) for t, v in car.items()])
    seasons = pd.DataFrame([
        {"driver_id": d, "year": y, "starts": 6, "wins": 6 if d == "a" else 0, "rating": r, "car": None,
         "h2h": np.nan, "champ_rank": rank}
        for y in (2013, 2014) for d, r, rank in (("a", 0.2, 1), ("b", 0.15, 2), ("c", 0.0, 3), ("d", -0.1, 4))
    ])
    team = {"a": "A", "b": "B", "c": "C", "d": "C"}
    seasons["car"] = [cars.set_index("team_season")["car"][f"{team[d]}_{y}"] for d, y in zip(seasons["driver_id"], seasons["year"])]
    regs = pd.DataFrame({"year": [2014], "kind": ["rules"], "change": ["Hybrid engines"]})
    return entries, seasons, cars, regs


def test_season_factors_car_rival_reliability_rules():
    f = season_factors(*_explain_inputs()).set_index(["driver_id", "year"])
    a = f.loc[("a", 2014)]
    assert a["car_rank"] == 1 and a["car_gap"] == pytest.approx(0.30)
    assert a["rival_id"] == "b" and not a["rival_same_car"]
    assert a["own_fail"] == 0 and a["rivals_fail"] > 0
    assert a["car_jump"] == pytest.approx(0.10)
    assert "2014: Hybrid engines" in a["rule_changes"]
    assert f.loc[("c", 2013), "teammate_id"] == "d" and f.loc[("c", 2013), "car_rank"] == 3
    assert f.loc[("a", 2013), "rule_changes"] == ""


def test_describe_labels():
    f = season_factors(*_explain_inputs())
    f["rival_name"], f["teammate_name"], f["team_name"] = f["rival_id"], f["teammate_id"], f["main_team"]
    res = describe(summarise_window(f.loc[(f["driver_id"] == "a") & (f["year"] == 2014)]), THR)
    cards = res["cards"]
    assert cards["Car"][0] == "Best car by a big margin"
    assert cards["Driver"][0] == "Best driver on the grid"
    assert cards["Competition"][0] == "Little competition"
    assert cards["Reliability"][0] == "Rivals broke down more"
    assert cards["New rules"][0] == "New rules — team nailed them"
    assert res["verdict"].startswith("Mostly the car")

    window = describe(summarise_window(f.loc[f["driver_id"] == "a"]), THR)
    assert window["cards"]["Car"][0] == "Best car by a big margin"  # best car in both seasons


def test_dashboard_and_pipeline_agree_on_data_version():
    """The app rebuilds when data_version differs, so both constants must move together."""
    import re
    from pathlib import Path

    from f1model.pipeline import DATA_VERSION

    app = (Path(__file__).resolve().parents[1] / "app" / "dashboard.py").read_text()
    assert int(re.search(r"^DATA_VERSION = (\d+)", app, re.M).group(1)) == DATA_VERSION

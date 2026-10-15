"""F1 Peak Dominance dashboard.

    python -m f1model.pipeline        # build data/out first
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Lets the app run on hosts that don't pip-install the package (e.g. Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1model.data.fetch import DATA_DIR  # noqa: E402
from f1model.metrics.explain import describe, summarise_window  # noqa: E402

OUT = DATA_DIR / "out"
REQUIRED = ("seasons", "peaks", "predictions", "track_strengths", "explain")
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GRID, MUTED = "rgba(137,135,129,0.25)", "#898781"
COMPONENTS = {
    "z_rating": "Skill vs car",
    "z_h2h": "Teammate H2H",
    "z_points_share": "Points share",
    "z_margin": "Margin over field",
}

st.set_page_config(page_title="F1 Peak Dominance", page_icon="🏁", layout="wide")


@st.cache_data
def load():
    data = {n: pd.read_parquet(OUT / f"{n}.parquet") for n in REQUIRED}
    bt = OUT / "backtest.parquet"
    data["backtest"] = pd.read_parquet(bt) if bt.exists() else None
    data["meta"] = json.loads((OUT / "meta.json").read_text())
    return data


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
        legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(font_size=13),
        bargap=0.25, bargroupgap=0.06,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID)
    return fig


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def rebuild(refresh: bool = False) -> None:
    """Run the pipeline with freshly imported code.

    After a new version is deployed, Streamlit re-runs this script but keeps
    already-imported f1model modules in memory, so an old pipeline could run.
    """
    for name in [m for m in sys.modules if m == "f1model" or m.startswith("f1model.")]:
        del sys.modules[name]
    from f1model import pipeline as fresh

    fresh.run(refresh=refresh)
    load.clear()


def outputs_ready() -> bool:
    return all((OUT / f"{n}.parquet").exists() for n in REQUIRED)


@st.cache_resource
def build_lock() -> threading.Lock:
    """One lock per server, so visitors arriving during a build wait instead of starting another."""
    return threading.Lock()


if not outputs_ready():
    # Fresh install, new server, or an app update that added outputs: build everything once.
    with st.spinner("Building the model with the latest code and data (about 30 seconds)..."):
        with build_lock():
            if not outputs_ready():
                rebuild()
        load.clear()
    if not outputs_ready():
        st.error("The model build did not finish. Please reboot the app (Manage app → ⋮ → Reboot app).")
        st.stop()

data = load()

seasons, peaks, preds, strengths = data["seasons"], data["peaks"], data["predictions"], data["track_strengths"]
st.sidebar.title("🏁 F1 Peak Dominance")
explain = data["explain"]
thr = data["meta"]["explain_thresholds"]
PAGES = ["All-time peak ranking", "Driver career", "Why so dominant?", "Head-to-head", "Track predictor"]
page = st.sidebar.radio("Page", PAGES, key="page")


def why(driver_id: str, start: int, end: int) -> dict:
    rows = explain.loc[(explain["driver_id"] == driver_id) & explain["year"].between(start, end)]
    return describe(summarise_window(rows), thr)
st.sidebar.caption(f"Data: F1DB, 1950 – {data['meta']['last_race']}  \n{data['meta']['starts']:,} race starts")
if st.sidebar.button("Update to latest results"):
    with st.spinner("Downloading latest F1 results and rebuilding..."):
        with build_lock():
            rebuild(refresh=True)
    st.rerun()
with st.sidebar.expander("How the score works"):
    st.markdown(
        "**Dominance Index** for each season = weighted mix of four stats, each scored against "
        "drivers of the same decade (z-score):\n"
        "- 40% **skill vs car**: model rating that removes car strength using teammate comparisons\n"
        "- 20% **teammate head-to-head**: quali + race battles won\n"
        "- 20% **points share**: modern 25-18-15 points as % of max\n"
        "- 20% **margin over field**: points lead over next-best driver\n\n"
        "**Peak** = best 3 consecutive seasons (5+ starts each)."
    )

# ------------------------------------------------------------------ ranking
if page == "All-time peak ranking":
    st.header("All-time peak dominance")
    st.caption("Each driver's best 3-season run. Higher = more dominant against their own era.")
    c1, c2, c3 = st.columns([2, 1, 1])
    lo, hi = c1.slider("Peak started between", 1950, int(seasons["year"].max()), (1950, int(seasons["year"].max())))
    top_n = c2.selectbox("Show top", [10, 25, 50, 100], index=1)
    short = c3.checkbox("Include careers under 3 seasons", value=False)
    p = peaks.loc[peaks["peak_start"].between(lo, hi)]
    if not short:
        p = p.loc[~p["short_career"]]
    p = p.head(top_n).reset_index(drop=True)
    p["rank"] = p.index + 1
    p["window"] = p["peak_start"].astype(str) + "–" + p["peak_end"].astype(str)

    fig = go.Figure(go.Bar(
        x=p["peak_score"], y=p["driver"] + "  (" + p["window"] + ")", orientation="h",
        marker=dict(color=BLUE, cornerradius=4),
        customdata=p[["peak_wins", "peak_titles", "peak_h2h", "peak_teams"]],
        hovertemplate="<b>%{y}</b><br>Index %{x:.2f}<br>Wins %{customdata[0]} · Titles %{customdata[1]}"
                      "<br>Teammate H2H %{customdata[2]:.0%}<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Peak Dominance Index")
    st.plotly_chart(style(fig, height=max(320, 26 * len(p))), use_container_width=True)

    st.dataframe(
        p[["rank", "driver", "window", "peak_score", "peak_rating", "peak_wins", "peak_poles", "peak_titles",
           "peak_h2h", "peak_car", "peak_teams", "peak_starts"]],
        hide_index=True, use_container_width=True,
        column_config={
            "rank": "#", "driver": "Driver", "window": "Peak",
            "peak_score": st.column_config.NumberColumn("Dominance", format="%.2f"),
            "peak_rating": st.column_config.NumberColumn("Skill rating", format="%.3f",
                help="Pure driver rating (car removed), in field-percentile units"),
            "peak_wins": "Wins", "peak_poles": "Poles", "peak_titles": "Titles*",
            "peak_h2h": st.column_config.NumberColumn("Teammate H2H", format="percent"),
            "peak_car": st.column_config.NumberColumn("Car rating", format="%.3f"),
            "peak_teams": "Teams", "peak_starts": "Starts",
        },
    )
    st.caption("*Titles under modern points re-scoring, so a few differ from history.")

# ------------------------------------------------------------------- career
elif page == "Driver career":
    names = peaks.sort_values("peak_score", ascending=False)["driver"].tolist()
    driver = st.selectbox("Driver", names, index=0)
    d = seasons.loc[seasons["driver"] == driver].sort_values("year")
    pk = peaks.loc[peaks["driver"] == driver].iloc[0]
    st.header(driver)
    m = st.columns(5)
    m[0].metric("Peak", f"{pk.peak_start}–{pk.peak_end}")
    m[1].metric("Peak index", f"{pk.peak_score:.2f}")
    m[2].metric("All-time rank", int(peaks.index[peaks["driver"] == driver][0]) + 1)
    m[3].metric("Career wins", int(d["wins"].sum()))
    m[4].metric("Teammate H2H (career)",
                pct((d["h2h_quali_w"].sum() + d["h2h_race_w"].sum()) / max(1, d["h2h_quali_n"].sum() + d["h2h_race_n"].sum())))

    st.info(f"**Why so dominant ({pk.peak_start}–{pk.peak_end})?** {why(pk.driver_id, pk.peak_start, pk.peak_end)['verdict']}. "
            "See the **Why so dominant?** page for the full breakdown.")

    st.subheader("Dominance Index by season")
    fig = go.Figure()
    fig.add_vrect(x0=pk.peak_start - 0.5, x1=pk.peak_end + 0.5, fillcolor=BLUE, opacity=0.08, line_width=0,
                  annotation_text="Peak", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=d["year"], y=d["dominance"], mode="lines+markers", line=dict(color=BLUE, width=2),
        marker=dict(size=8), customdata=d[["teams", "wins", "starts", "champ_rank"]], name="Dominance",
        hovertemplate="<b>%{x}</b> · %{customdata[0]}<br>Index %{y:.2f}<br>%{customdata[1]} wins / %{customdata[2]} starts"
                      "<br>Championship (re-scored): P%{customdata[3]}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    st.plotly_chart(style(fig, 340), use_container_width=True)

    st.subheader("Driver skill vs car strength")
    st.caption("Both in the same units: share of the field gained over an average driver / average car.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["year"], y=d["rating"], name="Driver skill", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=8),
                             hovertemplate="%{x}: skill %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["year"], y=d["car"], name="Car", mode="lines+markers",
                             line=dict(color=ORANGE, width=2), marker=dict(size=8, symbol="square"),
                             hovertemplate="%{x}: car %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    st.plotly_chart(style(fig, 320), use_container_width=True)

    st.dataframe(
        d[["year", "teams", "starts", "wins", "podiums", "poles", "points_share", "h2h", "tm_quali_gap",
           "rating", "car", "dominance"]].sort_values("year", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            "points_share": st.column_config.NumberColumn("Points share", format="percent"),
            "h2h": st.column_config.NumberColumn("Teammate H2H", format="percent"),
            "tm_quali_gap": st.column_config.NumberColumn("Quali gap to TM", format="%+.2f",
                help="Average qualifying percentile ahead of teammate"),
            "rating": st.column_config.NumberColumn("Skill", format="%+.3f"),
            "car": st.column_config.NumberColumn("Car", format="%+.3f"),
            "dominance": st.column_config.NumberColumn("Index", format="%.2f"),
        },
    )

# ------------------------------------------------------------------- why
elif page == "Why so dominant?":
    st.header("Why so dominant?")
    st.caption("Breaks a driver's results into car, driver, competition, reliability and rule changes.")
    names = peaks.sort_values("peak_score", ascending=False)["driver"].tolist()
    c1, c2 = st.columns([2, 1])
    driver = c1.selectbox("Driver", names, index=0)
    pk = peaks.loc[peaks["driver"] == driver].iloc[0]
    years = sorted(explain.loc[explain["driver_id"] == pk.driver_id, "year"].unique(), reverse=True)
    peak_label = f"Peak ({pk.peak_start}–{pk.peak_end})"
    period = c2.selectbox("Period", [peak_label] + [str(y) for y in years])
    start, end = (pk.peak_start, pk.peak_end) if period == peak_label else (int(period), int(period))

    rows = explain.loc[(explain["driver_id"] == pk.driver_id) & explain["year"].between(start, end)]
    f = summarise_window(rows)
    res = describe(f, thr)
    st.subheader(res["verdict"])

    share = res["driver_share"]
    if pd.notna(share):
        fig = go.Figure()
        for name, val, color in (("Driver", share, BLUE), ("Car", 1 - share, ORANGE)):
            fig.add_trace(go.Bar(
                y=["Advantage"], x=[val], name=name, orientation="h", marker=dict(color=color),
                text=f"{name} {val:.0%}", textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{name}: {val:.0%} of the advantage<extra></extra>",
            ))
        fig.update_layout(barmode="stack", showlegend=False)
        fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style(fig, 120), use_container_width=True)
        st.caption(f"Driver edge {f['driver_edge']:+.2f} and car edge {f['car_edge']:+.2f} over the average "
                   "driver and car that season (model estimate).")
    else:
        st.caption("Neither the driver nor the car was above average here, so there is no split to show.")

    cols = st.columns(5)
    for col, (factor, (label, detail)) in zip(cols, res["cards"].items()):
        with col.container(border=True):
            st.caption(factor.upper())
            st.markdown(f"**{label}**")
            st.caption(detail)

    with st.expander("Era context", expanded=True):
        e1, e2, e3 = st.columns(3)
        e1.markdown(f"**Team**  \n{f['team_name'] or '–'}")
        e2.markdown(f"**Engine**  \n{f['engine_id'].replace('-', ' ').title() or '–'}")
        e3.markdown(f"**Tyres**  \n{f['tyre_id'].replace('-', ' ').title() or '–'}")
        st.markdown(f"**Major rule changes around this time:** {f['rule_changes'] or 'none in the list'}")

    st.caption("Limits: the driver/car split is a model estimate. When a driver crushes a weak teammate, the model "
               "credits the driver more and the car less, so the teammate's championship finish is shown as a "
               "model-free clue. 'New rules' uses a short hand-written list (data/regulations.csv).")

# ------------------------------------------------------------- head-to-head
elif page == "Head-to-head":
    st.header("Compare two peaks")
    names = peaks.sort_values("peak_score", ascending=False)["driver"].tolist()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Driver A", names, index=0)
    b = c2.selectbox("Driver B", names, index=min(1, len(names) - 1))

    def peak_profile(name):
        pk = peaks.loc[peaks["driver"] == name].iloc[0]
        s = seasons.loc[(seasons["driver"] == name) & seasons["year"].between(pk.peak_start, pk.peak_end)]
        w = s["starts"]
        comp = {k: (s[k].fillna(0) * w).sum() / w.sum() for k in COMPONENTS}
        return pk, s, comp

    (pa, sa, ca), (pb, sb, cb) = peak_profile(a), peak_profile(b)
    labels = list(COMPONENTS.values())
    fig = go.Figure()
    for name, pk, comp, color in ((a, pa, ca, BLUE), (b, pb, cb, ORANGE)):
        fig.add_trace(go.Bar(
            name=f"{name} ({pk.peak_start}–{pk.peak_end})", x=labels, y=list(comp.values()),
            marker=dict(color=color, cornerradius=4), hovertemplate="%{x}: %{y:+.2f} σ<extra>" + name + "</extra>",
        ))
    fig.update_yaxes(title="Standard deviations above era average")
    st.plotly_chart(style(fig, 380), use_container_width=True)

    def row(pk, s):
        return {
            "Peak": f"{pk.peak_start}–{pk.peak_end}", "Dominance": round(pk.peak_score, 2),
            "Skill rating": round(pk.peak_rating, 3), "Car rating": round(pk.peak_car, 3),
            "Wins": int(s["wins"].sum()), "Starts": int(s["starts"].sum()),
            "Win rate": pct(s["wins"].sum() / s["starts"].sum()), "Poles": int(s["poles"].sum()),
            "Teammate H2H": pct(pk.peak_h2h), "Teams": pk.peak_teams,
        }

    st.dataframe(pd.DataFrame({a: row(pa, sa), b: row(pb, sb)}).astype(str), use_container_width=True)

# ---------------------------------------------------------------- predictor
else:
    st.header("Track predictor")
    st.caption("Predicted before qualifying, using the current grid, recent driver and car form, and each "
               "driver's history on this type of track and circuit.")
    races = (preds.drop_duplicates("race_id")
             .assign(label=lambda r: r.apply(
                 lambda x: f"{x['date']} · {x['circuit']}" if x["upcoming"] else f"What if: {x['circuit']}", axis=1))
             .sort_values(["upcoming", "date", "circuit"], ascending=[False, True, True]))
    choice = st.selectbox("Race", races["label"].tolist())
    rid = races.loc[races["label"] == choice, "race_id"].iat[0]
    r = preds.loc[preds["race_id"] == rid].sort_values("pred_rank")
    tt = r["track_type"].iat[0]
    st.markdown(f"**{r['circuit'].iat[0]}** · track type: **{tt}**")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Predicted order")
        st.dataframe(
            r[["pred_rank", "driver", "team", "p_win", "p_podium", "type_adj", "circuit_adj"]],
            hide_index=True, use_container_width=True, height=36 * len(r) + 40,
            column_config={
                "pred_rank": "#", "driver": "Driver", "team": "Team",
                "p_win": st.column_config.ProgressColumn("Win chance", format="percent", min_value=0, max_value=1),
                "p_podium": st.column_config.ProgressColumn("Podium chance", format="percent", min_value=0, max_value=1),
                "type_adj": st.column_config.NumberColumn(f"{tt} form", format="%+.3f",
                    help="Better (+) or worse (−) than usual on this track type"),
                "circuit_adj": st.column_config.NumberColumn("Circuit form", format="%+.3f",
                    help="Better (+) or worse (−) than usual at this exact circuit"),
            },
        )
    with right:
        st.subheader(f"Who gains on {tt} tracks")
        s = strengths.loc[strengths["track_type"] == tt].sort_values("type_adj")
        fig = go.Figure(go.Bar(
            x=s["type_adj"], y=s["driver"], orientation="h",
            marker=dict(color=[BLUE if v >= 0 else RED for v in s["type_adj"]], cornerradius=4),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
        fig.update_xaxes(title="vs own usual form (field share)")
        st.plotly_chart(style(fig, 36 * len(s) + 60), use_container_width=True)

    if data["backtest"] is not None:
        with st.expander("How accurate is this? (backtest)"):
            bt = data["backtest"]
            summary = bt.groupby("model").agg(races=("race_id", "nunique"), spearman=("spearman", "mean"),
                                               mae=("mae", "mean"), winner_hit=("winner_hit", "mean"))
            st.dataframe(
                summary.sort_values("spearman", ascending=False), use_container_width=True,
                column_config={
                    "spearman": st.column_config.NumberColumn("Rank accuracy", format="%.3f",
                        help="Spearman correlation of predicted vs actual order (1 = perfect)"),
                    "mae": st.column_config.NumberColumn("Avg places off", format="%.2f"),
                    "winner_hit": st.column_config.NumberColumn("Picked winner", format="percent"),
                },
            )
            st.caption("Each season predicted using only earlier seasons. Track-type effects are real but small: "
                       "most of what decides a race is car and current form.")

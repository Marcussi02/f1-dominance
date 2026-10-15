# F1 Peak Dominance

How dominant was each F1 driver at their peak, measured fairly across eras? And how will today's drivers do at upcoming tracks?

## Run it on your computer

Needs Python 3.11 or newer.

```bash
git clone https://github.com/Marcussi02/f1-dominance.git
cd f1-dominance
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
streamlit run app/dashboard.py     # opens http://localhost:8501
```

The first start downloads the F1 data and builds the model (about 30 seconds). After a race weekend, click **Update to latest results** in the sidebar.

Optional:
```bash
python -m f1model.pipeline --refresh   # rebuild from the command line and print the results
pytest                                 # unit tests
```

## Host it online (free, Streamlit Community Cloud)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Repository: `Marcussi02/f1-dominance` · Branch: `main` · Main file: `app/dashboard.py`.
4. Under **Advanced settings**, choose Python 3.12 or newer, then click **Deploy**.

You get a link like `https://<name>.streamlit.app`. Every push to `main` redeploys it automatically.

## How it works

**Data:** [F1DB](https://github.com/f1db/f1db), every championship race from 1950 to now. The Indy 500 and shared drives are left out.

**Driver vs car:** a ridge regression explains each race and qualifying result as *driver skill + car strength*. Teammates share the car, so the gap between them reveals skill. Drivers who change teams link the whole grid across eras. A "car" is entrant + constructor, so privateer customer cars are not counted as works teammates.

**Dominance Index (per season):** z-scored within each decade.

| Weight | Component |
|---|---|
| 40% | Skill vs car (model rating) |
| 20% | Teammate head-to-head (quali + race) |
| 20% | Points share (modern 25-18-15 points) |
| 20% | Margin over the next-best driver |

**Peak:** the best 3 consecutive seasons with 5+ starts each.

**Why so dominant?** For any driver's peak (or a single season), the app shows a verdict, a driver-vs-car split, and five factor cards:
- **Car:** rank on the grid, plus the teammate's championship finish as a model-free clue.
- **Driver:** skill rank and teammate head-to-head.
- **Competition:** the closest rival's driver + car package.
- **Reliability:** own car failures vs the top rivals'.
- **New rules:** major rule changes from `data/regulations.csv`, plus the team's car improvement that year.

Labels are compared against title-winning seasons, so they mean the same thing in every era.

**Track predictor:** recent driver form, car form, reliability, teammate edge, and how the driver does on this *track type* and *circuit* compared with their usual form. Track types (street / high-speed / mixed / technical) come from an era-adjusted pole speed. Edit `data/circuits.csv` to override any label.

## Backtest (2014–2026, 269 races, each season predicted from earlier seasons only)

| Model | Rank accuracy | Avg places off | Picked winner |
|---|---|---|---|
| Model, after qualifying | **0.662** | **3.38** | 55% |
| Grid order | 0.641 | 3.47 | 56% |
| Model, before qualifying | 0.617 | 3.69 | 45% |
| Recent form only | 0.618 | 3.71 | 43% |

Track-type effects are real but small. Car and current form decide most races.

## Limits
- Before 2018 there's no lap-time data, so ratings use positions only.
- Eras with few team changes (e.g. early 1950s) are less tightly linked to the rest of history.
- Points re-scoring ignores sprints and fastest-lap bonuses.

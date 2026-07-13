# 🏆 FIFA World Cup 2026 — Champion Predictor

Monte Carlo simulation of the World Cup 2026 semifinals and final, live data + real bookmaker
odds, calibrated with `scipy.optimize`.

Built by Soham Chatterjee.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

## What it does

- Pulls real tournament goals-for/against for the four semifinalists (France, Spain, England,
  Argentina) from [football-data.org](https://www.football-data.org/client/register).
- Pulls live bookmaker outright title odds from [The Odds API](https://the-odds-api.com/#get-access),
  via the dedicated `soccer_fifa_world_cup_winner` sport key (outright markets live under their
  own `_winner` key, separate from the regular match-odds key).
- Builds a power rating per team: 30% tournament goal form (winsorized so a small-sample
  shutout defense can't dominate), 70% market-implied strength.
- Calibrates a logistic win-probability scale against real bookmaker odds.
- Simulates the semifinals + final up to 50,000 times for a champion probability per team.
- Falls back to a cached July 12, 2026 snapshot for anything an API call doesn't return, and
  always shows which source was actually used.

## Bracket (confirmed July 12, 2026)

| Match | Date | Venue |
|---|---|---|
| SF1: France vs Spain | Jul 14 | Dallas |
| SF2: England vs Argentina | Jul 15 | Atlanta |
| Final | Jul 19 | New Jersey (MetLife Stadium) |

## Project structure

```
wc2026-champion-predictor/
├── app.py                        # Streamlit app (entry point)
├── src/
│   ├── api_client.py              # live API calls + graceful fallback
│   ├── model.py                   # power ratings + market calibration
│   └── simulate.py                # Monte Carlo bracket simulation
├── data/
│   ├── team_stats_fallback.csv    # cached snapshot (goals, odds, pedigree)
│   └── fixtures.csv               # bracket / results so far
├── requirements.txt
└── .streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your free API keys in the sidebar, or leave them blank to run on the cached snapshot.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), "New app".
3. Point it at this repo, branch `main`, file `app.py`. Deploy.

## API keys

Both free, no card required:
- football-data.org: https://www.football-data.org/client/register
- The Odds API: https://the-odds-api.com/#get-access

## Notes / limitations

Not trying to beat the bookmakers — this reproduces their thinking using goal stats + real odds
+ some probability math, calibrated with scipy. Good for a rough estimate, not a lock. Once the
final matchup is known, hardcode `sf1`/`sf2` in `simulate_tournament` and it collapses to a
single-match probability instead of a bracket.

---
Built by Soham Chatterjee · [github.com/VibeCipher](https://github.com/VibeCipher)

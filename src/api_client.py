"""
api_client.py
Pulls live data for the WC 2026 semifinalists:
  - football-data.org  -> match results (goals for/against)
  - The Odds API        -> title odds, via the dedicated outrights sport key
                            'soccer_fifa_world_cup_winner' (NOT the regular
                            match-odds key 'soccer_fifa_world_cup' -- that one
                            doesn't support the outrights market and 422s).

Both calls degrade gracefully: on any failure (missing key, network error,
no matching data) they return None and the caller falls back to a cached
snapshot instead of crashing.
"""

import pandas as pd
import requests

SEMIFINALISTS = ["France", "Spain", "England", "Argentina"]


def fetch_wc_matches(token: str) -> dict:
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    headers = {"X-Auth-Token": token}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_football_data(payload: dict, teams=SEMIFINALISTS):
    rows = []
    for m in payload.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        h, a = m["homeTeam"]["name"], m["awayTeam"]["name"]
        full = m.get("score", {}).get("fullTime", {})
        hs, aws = full.get("home"), full.get("away")
        if hs is None or aws is None:
            continue
        if h in teams:
            rows.append({"team": h, "goals_for": hs, "goals_against": aws})
        if a in teams:
            rows.append({"team": a, "goals_for": aws, "goals_against": hs})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.groupby("team").agg(
        games_played=("team", "count"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    ).reset_index()


def get_live_goals(token: str):
    if not token:
        return None, "No football-data.org token provided"
    try:
        payload = fetch_wc_matches(token)
        df = parse_football_data(payload)
        if df is None:
            return None, "football-data.org returned no finished matches for these teams"
        return df, f"Pulled {len(payload.get('matches', []))} matches from football-data.org"
    except Exception as e:
        return None, f"football-data.org request failed ({e})"


def fetch_outright_odds(api_key: str) -> list:
    # Outright/futures markets live under a dedicated '_winner' sport key,
    # separate from the regular match-odds sport key.
    url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds"
    params = {"apiKey": api_key, "regions": "us,uk,eu", "markets": "outrights", "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_outrights(payload: list, teams=SEMIFINALISTS,
                     preferred=("fanduel", "draftkings", "pinnacle", "bet365")):
    for event in payload:
        bms = {bm["key"]: bm for bm in event.get("bookmakers", [])}
        bm = next((bms[k] for k in preferred if k in bms), next(iter(event.get("bookmakers", [])), None))
        if not bm:
            continue
        for market in bm.get("markets", []):
            if market["key"] == "outrights":
                df = pd.DataFrame(market["outcomes"]).rename(columns={"name": "team", "price": "odds_decimal"})
                df = df[df["team"].isin(teams)]
                if not df.empty:
                    return df
    return None


def decimal_to_american(dec: float) -> int:
    dec = float(dec)
    return round((dec - 1) * 100) if dec >= 2.0 else round(-100 / (dec - 1))


def get_live_odds(api_key: str):
    if not api_key:
        return None, "No Odds API key provided"
    try:
        payload = fetch_outright_odds(api_key)
        df = parse_outrights(payload)
        if df is None:
            return None, "No outrights market found for these teams"
        df = df.copy()
        df["odds_to_win_american"] = df["odds_decimal"].apply(decimal_to_american)
        return df[["team", "odds_to_win_american"]], "Pulled LIVE outright odds from The Odds API"
    except Exception as e:
        return None, f"The Odds API request failed ({e})"

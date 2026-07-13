"""
model.py
Power rating + market calibration for the WC 2026 final four.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

TEAMS = ["France", "Spain", "England", "Argentina"]

FIFA_PEDIGREE = {"France": 2, "Spain": 1, "England": 1, "Argentina": 3}


def implied_prob_from_american(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def build_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blends two signals per team:
      - stat_power: attack/defense index from real tournament goals,
        winsorized so a small-sample shutout defense can't dominate.
      - market_power: logit of normalized bookmaker win probability --
        captures injuries, fatigue, tactics that goal counts alone can't see.
    Final power = 0.3 * stat_power (z-scored) + 0.7 * market_power (z-scored).
    """
    df = df.copy()
    df["attack_rate"] = df["goals_for"] / df["games_played"]
    df["defense_rate"] = df["goals_against"] / df["games_played"]
    df["attack_idx"] = df["attack_rate"] / df["attack_rate"].mean()
    raw_defense_idx = df["defense_rate"] / df["defense_rate"].mean()
    df["defense_idx"] = raw_defense_idx.clip(lower=0.35, upper=2.5)

    df["stat_power"] = 40 * df["attack_idx"] - 25 * df["defense_idx"] + 3 * df["fifa_pedigree_titles"]

    market_prob = df["odds_to_win_american"].apply(implied_prob_from_american)
    norm_prob = market_prob / market_prob.sum()
    eps = 1e-6
    df["market_power"] = np.log(norm_prob.clip(eps, 1 - eps) / (1 - norm_prob.clip(eps, 1 - eps)))

    def zscore(s):
        return (s - s.mean()) / s.std()

    df["power"] = 0.3 * zscore(df["stat_power"]) + 0.7 * zscore(df["market_power"])
    return df


def win_prob(rating_a, rating_b, scale):
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / scale))


def calibrate_scale(ratings: pd.DataFrame) -> float:
    """Fits the logistic scale so the model's analytic title probabilities
    (given each team's likely bracket path) line up with real bookmaker odds."""
    rt = ratings.set_index("team")["power"]
    bracket_paths = {
        "France":    [rt["Spain"], (rt["England"] + rt["Argentina"]) / 2],
        "Spain":     [rt["France"], (rt["England"] + rt["Argentina"]) / 2],
        "England":   [rt["Argentina"], (rt["France"] + rt["Spain"]) / 2],
        "Argentina": [rt["England"], (rt["France"] + rt["Spain"]) / 2],
    }
    market = ratings.copy()
    market["market_prob"] = market["odds_to_win_american"].apply(implied_prob_from_american)
    market["market_prob"] = market["market_prob"] / market["market_prob"].sum()

    def analytic_title_probs(scale):
        probs = {}
        for team, rounds in bracket_paths.items():
            p = 1.0
            for opp_rating in rounds:
                p *= win_prob(rt[team], opp_rating, scale)
            probs[team] = p
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def loss(scale):
        sim_probs = analytic_title_probs(scale)
        return sum((sim_probs[row.team] - row.market_prob) ** 2 for row in market.itertuples())

    return minimize_scalar(loss, bounds=(5, 200), method="bounded").x

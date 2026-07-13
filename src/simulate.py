"""
simulate.py
Monte Carlo simulation of the WC 2026 semifinals + final.

  SF1: France vs Spain           -> Dallas, Jul 14
  SF2: England vs Argentina      -> Atlanta, Jul 15
  Final: SF1 winner vs SF2 winner -> New Jersey, Jul 19
"""

import numpy as np
import pandas as pd

from model import win_prob

AVG_GOALS = 2.3


def simulate_match(team_a, team_b, ratings_indexed, scale, rng):
    ra = ratings_indexed.loc[team_a, "power"]
    rb = ratings_indexed.loc[team_b, "power"]
    return team_a if rng.random() < win_prob(ra, rb, scale) else team_b


def cosmetic_scoreline(team_a, team_b, winner, ratings_indexed, rng):
    ra, rb = ratings_indexed.loc[team_a], ratings_indexed.loc[team_b]
    lam_a = max(0.4, AVG_GOALS / 2 * ra["attack_idx"] * rb["defense_idx"])
    lam_b = max(0.4, AVG_GOALS / 2 * rb["attack_idx"] * ra["defense_idx"])
    ga, gb = rng.poisson(lam_a), rng.poisson(lam_b)
    if winner == team_a and ga <= gb:
        ga = gb + 1
    elif winner == team_b and gb <= ga:
        gb = ga + 1
    return int(ga), int(gb)


def simulate_tournament(ratings_df, scale, n_sims=30000, seed=42,
                         sf1=("France", "Spain"), sf2=("England", "Argentina")):
    rng = np.random.default_rng(seed)
    rdf = ratings_df.set_index("team")
    champions, finalists_log = [], []
    for _ in range(n_sims):
        sf1_w = simulate_match(sf1[0], sf1[1], rdf, scale, rng)
        sf2_w = simulate_match(sf2[0], sf2[1], rdf, scale, rng)
        finalists_log.extend([sf1_w, sf2_w])
        champions.append(simulate_match(sf1_w, sf2_w, rdf, scale, rng))
    return {
        "champion_probs": (pd.Series(champions).value_counts() / n_sims).sort_values(ascending=False),
        "finalist_probs": (pd.Series(finalists_log).value_counts() / n_sims).sort_values(ascending=False),
        "n_sims": n_sims,
    }


def simulate_one_bracket(ratings_df, scale, seed=None,
                          sf1=("France", "Spain"), sf2=("England", "Argentina")):
    rng = np.random.default_rng(seed)
    rdf = ratings_df.set_index("team")

    sf1_w = simulate_match(*sf1, rdf, scale, rng)
    sf1_s = cosmetic_scoreline(*sf1, sf1_w, rdf, rng)

    sf2_w = simulate_match(*sf2, rdf, scale, rng)
    sf2_s = cosmetic_scoreline(*sf2, sf2_w, rdf, rng)

    champ = simulate_match(sf1_w, sf2_w, rdf, scale, rng)
    final_s = cosmetic_scoreline(sf1_w, sf2_w, champ, rdf, rng)

    return {
        "SF1": (sf1, sf1_s, sf1_w),
        "SF2": (sf2, sf2_s, sf2_w),
        "Final": ((sf1_w, sf2_w), final_s, champ),
        "champion": champ,
    }

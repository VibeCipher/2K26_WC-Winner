import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import SEMIFINALISTS, get_live_goals, get_live_odds
from model import FIFA_PEDIGREE, build_ratings, calibrate_scale, implied_prob_from_american
from simulate import simulate_one_bracket, simulate_tournament

st.set_page_config(page_title="WC 2026 Champion Predictor", page_icon="🏆", layout="wide")

st.markdown("""
<style>
/* Football pitch backdrop -- deep stadium green with faint pitch stripes + center circle glow */
.stApp {
    background:
        radial-gradient(circle at 50% 42%, rgba(255,255,255,0.05) 0px, rgba(255,255,255,0.05) 1px, transparent 1px),
        repeating-linear-gradient(
            0deg,
            #163a1e 0px, #163a1e 90px,
            #1a4322 90px, #1a4322 180px
        );
    background-size: auto, 100% 180px;
}

/* Center-circle + halfway-line watermark, fixed so it doesn't scroll oddly */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 50% 50%, transparent 0, transparent 130px, rgba(255,255,255,0.06) 132px, rgba(255,255,255,0.06) 134px, transparent 136px),
        linear-gradient(90deg, transparent calc(50% - 1px), rgba(255,255,255,0.06) calc(50% - 1px), rgba(255,255,255,0.06) calc(50% + 1px), transparent calc(50% + 1px));
    z-index: 0;
}

/* Frosted cards for readability over the pitch */
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]) {
    position: relative;
    z-index: 1;
}
div.block-container {
    position: relative;
    z-index: 1;
    max-width: 1150px;
    background: rgba(240, 246, 240, 0.93);
    border-top: 6px solid #2e7d32;
    border-radius: 16px;
    padding: 2rem 2.5rem 2.5rem;
    margin: 1.5rem auto 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.45);
}

/* Title flourish */
h1 {
    text-shadow: 0 1px 0 rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FLAGS = {"France": "🇫🇷", "Spain": "🇪🇸", "Argentina": "🇦🇷", "England": "🏴"}
TEAM_COLORS = {
    "France": "#0055A4",
    "Argentina": "#75AADB",
    "Spain": "#C60B1E",
    "England": "#FFFFFF",
}


def load_fallback_stats():
    return pd.read_csv(os.path.join(DATA_DIR, "team_stats_fallback.csv"))


@st.cache_data(ttl=900, show_spinner=False)
def get_team_stats(football_token: str, odds_key: str):
    fallback = load_fallback_stats()
    status_msgs = []

    goals_df, goals_msg = get_live_goals(football_token)
    status_msgs.append(("goals", goals_df is not None, goals_msg))
    if goals_df is None:
        goals_df = fallback[["team", "games_played", "goals_for", "goals_against"]]

    odds_df, odds_msg = get_live_odds(odds_key)
    status_msgs.append(("odds", odds_df is not None, odds_msg))
    if odds_df is None:
        odds_df = fallback[["team", "odds_to_win_american"]]

    team_stats = goals_df.merge(odds_df, on="team", how="left")
    team_stats["fifa_pedigree_titles"] = team_stats["team"].map(FIFA_PEDIGREE)
    return team_stats, status_msgs


st.title("🏆 FIFA World Cup 2026 — Champion Predictor")
st.caption(
    "Built by Soham Chatterjee. Monte Carlo simulation of the semifinals + final, "
    "calibrated against real bookmaker odds. Data pulled live from football-data.org "
    "and The Odds API where available, cached fallback otherwise."
)

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


football_token = get_secret("FOOTBALL_DATA_TOKEN")
odds_key = get_secret("ODDS_API_KEY")

team_stats, status_msgs = get_team_stats(football_token, odds_key)

with st.expander("Data source status"):
    for kind, is_live, msg in status_msgs:
        if is_live:
            st.success(msg)
        else:
            st.warning(f"{msg} -- using cached snapshot")

ratings = build_ratings(team_stats)
scale = calibrate_scale(ratings)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "🎲 Random Outcome", "📈 Team Ratings", "ℹ️ How this works"])

# ---------------------------------------------------------------------------
with tab1:
    n_sims = st.slider("Number of Monte Carlo simulations", 1000, 50000, 30000, step=1000)
    with st.spinner("Running simulations..."):
        result = simulate_tournament(ratings, scale, n_sims=n_sims)

    champ_probs = result["champion_probs"]

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Probability of Winning the Tournament")
        chart_df = pd.DataFrame({
            "Team": champ_probs.index,
            "Probability (%)": (champ_probs.values * 100).round(1),
        })
        fig = px.bar(
            chart_df, x="Probability (%)", y="Team", orientation="h",
            text="Probability (%)", color="Team", color_discrete_map=TEAM_COLORS,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                           marker_line_color="#333333", marker_line_width=1.2)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model vs. Market")
        market = ratings.copy()
        market["market_prob"] = market["odds_to_win_american"].apply(implied_prob_from_american)
        market["market_prob"] = market["market_prob"] / market["market_prob"].sum() * 100
        market["model_prob"] = market["team"].map(lambda t: champ_probs.get(t, 0) * 100)
        cmp_df = market[["team", "model_prob", "market_prob"]].sort_values("model_prob", ascending=False)
        cmp_df.columns = ["Team", "Model %", "Bookmaker %"]
        st.dataframe(cmp_df.set_index("Team").round(1), use_container_width=True)
        st.caption(f"Calibrated logistic scale: {scale:.2f}")

    st.divider()
    st.subheader("Semifinal / Final")
    st.markdown("**SF1:** France vs Spain — Dallas, Jul 14")
    st.markdown("**SF2:** England vs Argentina — Atlanta, Jul 15")
    st.markdown("**Finalist probability**")
    f_df = (result["finalist_probs"] * 100).round(1).rename("Probability (%)")
    st.dataframe(f_df, use_container_width=True)

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Play out one random version of the tournament")
    st.caption(
        "This is a single random draw, not the model's prediction -- check the Predictions tab "
        "for the actual probabilities. An underdog will win this some of the time, that's expected."
    )
    if st.button("🎲 Simulate one random outcome", type="primary"):
        outcome = simulate_one_bracket(ratings, scale, seed=np.random.randint(0, 1_000_000))

        def show_match(label, data):
            (t1, t2), (s1, s2), winner = data
            st.markdown(
                f"**{label}:** {FLAGS.get(t1, '')} {t1} **{s1} – {s2}** {t2} {FLAGS.get(t2, '')} "
                f"&nbsp;&nbsp;→&nbsp;&nbsp; 🏅 **{winner}** advances"
            )

        show_match("Semifinal 1 — France vs Spain", outcome["SF1"])
        show_match("Semifinal 2 — England vs Argentina", outcome["SF2"])
        st.markdown("---")
        show_match("🏆 FINAL", outcome["Final"])
        st.success(f"### {FLAGS.get(outcome['champion'], '')} {outcome['champion']} wins the 2026 FIFA World Cup!")
    else:
        st.info("Click the button to play out one full random bracket.")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Team Power Ratings")
    show_cols = ["team", "games_played", "goals_for", "goals_against",
                 "attack_idx", "defense_idx", "power", "odds_to_win_american"]
    display = ratings[show_cols].copy()
    display.columns = ["Team", "Games", "GF", "GA", "Attack Index", "Defense Index",
                        "Power Rating", "Odds to Win (American)"]
    display = display.sort_values("Power Rating", ascending=False)
    st.dataframe(display.set_index("Team").round(3), use_container_width=True)

    fig2 = px.scatter(
        ratings, x="attack_idx", y="defense_idx", text="team",
        color="team", color_discrete_map=TEAM_COLORS,
        labels={"attack_idx": "Attack Index (higher = more goals scored)",
                "defense_idx": "Defense Index (lower = fewer goals conceded)"},
    )
    fig2.update_traces(textposition="top center", marker=dict(size=22, line=dict(color="#333", width=1.2)))
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(height=460, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
with tab4:
    st.markdown(f"""
### how this works

not trying to beat the bookmakers here — this is basically trying to reproduce their thinking
using goal stats + real odds + some probability math, calibrated with scipy.

**data**
- goals: [football-data.org](https://www.football-data.org/client/register), competition `WC`
- odds: [The Odds API](https://the-odds-api.com/#get-access), sport key `soccer_fifa_world_cup_winner`
  (outright markets live under a dedicated `_winner` key, separate from the regular match-odds key)
- if a key's missing or a call fails, falls back to a snapshot from July 12, 2026 -- the
  "Data source status" expander at the top always shows which source was actually used

**power rating**
each team gets an attack/defense index from real tournament goals (capped so a small-sample
shutout defense can't dominate), blended 30/70 with a market-implied rating from bookmaker odds.

**calibration**
a single logistic scale constant is fit with `scipy.optimize.minimize_scalar` so the model's
title probabilities track real bookmaker odds as closely as possible. Current value: **{scale:.2f}**

**simulation**
the semifinals and final are played out {n_sims:,} times; each match winner is drawn from the
same calibrated logistic function used above.

**bracket**
- SF1: France vs Spain — Dallas, Jul 14
- SF2: England vs Argentina — Atlanta, Jul 15
- Final — New Jersey, Jul 19

not a lock, just a rough estimate.
""")

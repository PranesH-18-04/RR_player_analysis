"""
Second-Innings Win Probability Model for T20 Cricket (Chasing Side)
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis

This script:
1. Loads match and ball-by-ball records from 2022 onward.
2. Formulates game states (runs needed, balls remaining, wickets in hand, RRR).
3. Trains a Logistic Regression model on 2022-2024 matches.
4. Evaluates log loss, Brier score, and calibration on unseen 2025+ matches.
5. Computes delivery-level win probability swing (|P_after - P_before|).
6. Determines the 85th percentile swing threshold (top 15%) on training data.
7. Plots sample win probability trajectories and saves to data/win_prob_curves.png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

def predict_win_prob_vectorized(runs_needed, balls_rem, wkts, model):
    """Vectorized win probability evaluation with boundary/terminal conditions."""
    prob = np.zeros(len(runs_needed), dtype=float)
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_rem = np.asarray(balls_rem, dtype=float)
    wkts = np.asarray(wkts, dtype=float)

    # Terminal conditions:
    won_mask = runs_needed <= 0
    lost_mask = (balls_rem <= 0) | (wkts <= 0)
    in_play = ~won_mask & ~lost_mask

    prob[won_mask] = 1.0
    prob[lost_mask & ~won_mask] = 0.0

    if in_play.sum() > 0:
        rn = runs_needed[in_play]
        br = balls_rem[in_play]
        w = wkts[in_play]
        rrr = (rn * 6.0) / br
        X = pd.DataFrame({
            "runs_needed": rn,
            "balls_remaining": br,
            "wickets_in_hand": w,
            "rrr": rrr
        })
        prob[in_play] = model.predict_proba(X)[:, 1]

    return prob

def train_win_probability_model(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    os.makedirs(output_dir, exist_ok=True)

    matches_path = os.path.join(data_dir, "matches.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")

    print("===================================================================")
    print("WIN PROBABILITY MODEL PIPELINE")
    print("===================================================================")
    print(f"Loading matches from: {matches_path}")
    matches = pd.read_csv(matches_path)
    print(f"Loading ball_by_ball from: {bbb_path}")
    bbb = pd.read_csv(bbb_path)

    # 1. Clean matches: exclude no-result, tied, or abandoned
    drop_mask_matches = (
        matches["win_team_id"].isna()
        | (matches["win_team_id"] == 0)
        | matches["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    matches_valid = matches[~drop_mask_matches].copy()
    print(f"Matches kept: {len(matches_valid)} | Dropped: {drop_mask_matches.sum()}")

    # 2. Innings 1 total & Innings 2 target
    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    matches_valid = matches_valid.merge(inn1_totals, on="match_id", how="inner")

    # D/L Target handling
    matches_valid["target"] = np.where(
        matches_valid["revised_target"].notna() & (matches_valid["revised_target"] > 0),
        matches_valid["revised_target"],
        matches_valid["inn1_total"] + 1,
    )

    # Total balls allowed
    matches_valid["total_balls"] = np.where(
        matches_valid["second_inn_revised_overs"].notna() & (matches_valid["second_inn_revised_overs"] > 0),
        matches_valid["second_inn_revised_overs"],
        np.where(
            matches_valid["first_inn_revised_overs"].notna() & (matches_valid["first_inn_revised_overs"] > 0),
            matches_valid["first_inn_revised_overs"],
            120,
        ),
    )

    # 3. Innings 2 delivery records
    bbb_inn2 = bbb[bbb["innings"] == 2].copy()
    match_cols = ["match_id", "match_date", "win_team_id", "target", "total_balls", "home_team", "away_team", "result"]
    df = bbb_inn2.merge(matches_valid[match_cols], on="match_id", how="inner")
    df = df.sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)

    # State variables before and after each delivery
    df["cum_runs_after"] = df.groupby("match_id")["total_ball_runs"].cumsum()
    df["cum_runs_before"] = df["cum_runs_after"] - df["total_ball_runs"]

    df["cum_wickets_after"] = df.groupby("match_id")["is_wicket"].cumsum()
    df["cum_wickets_before"] = df["cum_wickets_after"] - df["is_wicket"].astype(int)

    df["balls_remaining_before"] = df["total_balls"] - (df["innings_ball_num"] - 1)
    df["balls_remaining_after"] = df["total_balls"] - df["innings_ball_num"]

    df["runs_needed_before"] = df["target"] - df["cum_runs_before"]
    df["runs_needed_after"] = df["target"] - df["cum_runs_after"]

    df["wickets_in_hand_before"] = 10 - df["cum_wickets_before"]
    df["wickets_in_hand_after"] = 10 - df["cum_wickets_after"]

    df["chase_win"] = (df["batting_team_id"] == df["win_team_id"]).astype(int)
    df["year"] = pd.to_datetime(df["match_date"]).dt.year

    # 4. Training on 2022-2024
    train_mask = (df["year"] <= 2024) & (df["balls_remaining_after"] > 0)
    train_df = df[train_mask].copy()
    train_df["rrr"] = (train_df["runs_needed_after"] * 6.0) / train_df["balls_remaining_after"]

    features = ["runs_needed", "balls_remaining", "wickets_in_hand", "rrr"]
    X_train = train_df[["runs_needed_after", "balls_remaining_after", "wickets_in_hand_after", "rrr"]].rename(
        columns={
            "runs_needed_after": "runs_needed",
            "balls_remaining_after": "balls_remaining",
            "wickets_in_hand_after": "wickets_in_hand"
        }
    )
    y_train = train_df["chase_win"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Coefficients ---")
    for f, c in zip(features, model.coef_[0]):
        print(f"  {f:18s}: {c:+.4f}")
    print(f"  Intercept         : {model.intercept_[0]:+.4f}")

    # 5. Out-of-sample Evaluation (2025+)
    test_mask = (df["year"] >= 2025) & (df["balls_remaining_after"] > 0)
    test_df = df[test_mask].copy()
    test_df["rrr"] = (test_df["runs_needed_after"] * 6.0) / test_df["balls_remaining_after"]
    X_test = test_df[["runs_needed_after", "balls_remaining_after", "wickets_in_hand_after", "rrr"]].rename(
        columns={
            "runs_needed_after": "runs_needed",
            "balls_remaining_after": "balls_remaining",
            "wickets_in_hand_after": "wickets_in_hand"
        }
    )
    y_test = test_df["chase_win"]

    test_preds = model.predict_proba(X_test)[:, 1]
    test_log_loss = log_loss(y_test, test_preds)
    test_brier = brier_score_loss(y_test, test_preds)

    print(f"\n--- Validation Metrics on Unseen Test Set (2025+) ---")
    print(f"Log Loss:    {test_log_loss:.4f}  (Uninformative baseline: 0.6931)")
    print(f"Brier Score: {test_brier:.4f}")

    # Calibration table (10 bins)
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    test_df["win_prob"] = test_preds
    test_df["prob_bin"] = pd.cut(test_df["win_prob"], bins=bins, labels=labels, include_lowest=True)

    calib = test_df.groupby("prob_bin", observed=False).agg(
        ball_count=("win_prob", "count"),
        mean_pred_prob=("win_prob", "mean"),
        actual_win_rate=("chase_win", "mean")
    ).reset_index()

    print(f"\n--- 10-Bin Calibration Table ---")
    print(calib.to_string(index=False))

    # 6. Compute Win Probability Swing for every ball in Innings 2
    p_before = predict_win_prob_vectorized(
        df["runs_needed_before"].values,
        df["balls_remaining_before"].values,
        df["wickets_in_hand_before"].values,
        model
    )
    p_after = predict_win_prob_vectorized(
        df["runs_needed_after"].values,
        df["balls_remaining_after"].values,
        df["wickets_in_hand_after"].values,
        model
    )

    df["p_before"] = p_before
    df["p_after"] = p_after
    df["wp_swing"] = np.abs(p_after - p_before)

    # 85th percentile threshold on training set (2022-2024)
    train_swings = df[df["year"] <= 2024]["wp_swing"]
    thresh_85 = float(np.percentile(train_swings, 85))

    print(f"\n--- Win Probability Swing Threshold (Task 1) ---")
    print(f"Training set deliveries: {len(train_swings)}")
    print(f"Top 15% Swing Threshold (85th percentile): {thresh_85:.5f} ({thresh_85*100:.2f}% swing)")

    # 7. Generate and save curves
    sample_ids = [40036, 40037, 40038, 40040]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, mid in enumerate(sample_ids):
        m_data = df[df["match_id"] == mid].sort_values("innings_ball_num")
        if m_data.empty:
            continue
        ax = axes[idx]
        ax.plot(m_data["innings_ball_num"], m_data["p_after"], color="#1A3C8F", lw=2.2, label="Chasing Win Prob")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="50% Equilibrium")

        # Mark wickets
        wkts = m_data[m_data["is_wicket"] == True]
        ax.scatter(wkts["innings_ball_num"], wkts["p_after"], color="#E91E8C", s=50, zorder=5, label="Wicket")

        # Highlight high-pressure balls (swing >= threshold)
        press_balls = m_data[m_data["wp_swing"] >= thresh_85]
        ax.scatter(press_balls["innings_ball_num"], press_balls["p_after"], facecolors="none", edgecolors="#E91E8C", s=80, lw=1.5, zorder=4, label="High Pressure Ball")

        info = m_data.iloc[0]
        ax.set_title(f"Match {mid}: {info['home_team']} vs {info['away_team']}\n{info['result']}", fontsize=9)
        ax.set_xlabel("Ball Number (Innings 2)")
        ax.set_ylabel("Win Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "win_prob_curves.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Saved validation curves to: {chart_path}")

    return model, thresh_85

def get_trained_model_and_threshold(data_dir=None):
    """Utility to obtain fitted model and 85th percentile swing threshold."""
    model, thresh_85 = train_win_probability_model(data_dir=data_dir)
    return model, thresh_85

if __name__ == "__main__":
    train_win_probability_model()

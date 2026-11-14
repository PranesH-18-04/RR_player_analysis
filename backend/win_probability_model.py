import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

def train_win_probability_model(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    matches_path = os.path.join(data_dir, "matches.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")

    print(f"Loading matches from {matches_path}")
    matches = pd.read_csv(matches_path)
    print(f"Loading ball_by_ball from {bbb_path}")
    bbb = pd.read_csv(bbb_path)

    # 1. Clean matches: exclude no-result, tied, or abandoned
    drop_mask_matches = (
        matches["win_team_id"].isna()
        | (matches["win_team_id"] == 0)
        | matches["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    matches_valid = matches[~drop_mask_matches].copy()
    print(f"Matches kept: {len(matches_valid)}, dropped: {drop_mask_matches.sum()}")

    # 2. Compute Innings 1 score & target
    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    matches_valid = matches_valid.merge(inn1_totals, on="match_id", how="inner")

    # Target: revised_target when D/L affected, otherwise Innings 1 total + 1
    matches_valid["target"] = np.where(
        matches_valid["revised_target"].notna() & (matches_valid["revised_target"] > 0),
        matches_valid["revised_target"],
        matches_valid["inn1_total"] + 1,
    )

    # Total balls allowed: second_inn_revised_overs if present, else first_inn_revised_overs, else 120
    matches_valid["total_balls"] = np.where(
        matches_valid["second_inn_revised_overs"].notna() & (matches_valid["second_inn_revised_overs"] > 0),
        matches_valid["second_inn_revised_overs"],
        np.where(
            matches_valid["first_inn_revised_overs"].notna() & (matches_valid["first_inn_revised_overs"] > 0),
            matches_valid["first_inn_revised_overs"],
            120,
        ),
    )

    # 3. Innings 2 ball-by-ball state
    bbb_inn2 = bbb[bbb["innings"] == 2].copy()
    match_cols = ["match_id", "match_date", "win_team_id", "target", "total_balls", "home_team", "away_team", "result"]
    df = bbb_inn2.merge(matches_valid[match_cols], on="match_id", how="inner")
    df = df.sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)

    # State variables
    df["cum_runs"] = df.groupby("match_id")["total_ball_runs"].cumsum()
    df["runs_needed"] = df["target"] - df["cum_runs"]
    df["balls_remaining"] = df["total_balls"] - df["innings_ball_num"]
    df["wickets_in_hand"] = 10 - df.groupby("match_id")["is_wicket"].cumsum()

    # RRR & CRR
    df["rrr"] = np.where(df["balls_remaining"] > 0, (df["runs_needed"] * 6.0) / df["balls_remaining"], np.nan)
    df["crr"] = np.where(df["innings_ball_num"] > 0, (df["cum_runs"] * 6.0) / df["innings_ball_num"], 0.0)

    # Chasing side win indicator
    df["chase_win"] = (df["batting_team_id"] == df["win_team_id"]).astype(int)

    # Drop balls where balls_remaining <= 0 (match concluded or over limit reached)
    invalid_mask = df["balls_remaining"] <= 0
    df_clean = df[~invalid_mask].copy()
    print(f"Deliveries modeled: {len(df_clean)}, dropped with balls_remaining <= 0: {invalid_mask.sum()}")

    # 4. Grouped Time-Based Train / Test Split
    df_clean["year"] = pd.to_datetime(df_clean["match_date"]).dt.year
    train_df = df_clean[df_clean["year"] <= 2024].copy()
    test_df = df_clean[df_clean["year"] >= 2025].copy()

    print(f"Train set (2022-2024): {train_df['match_id'].nunique()} matches | {len(train_df)} balls")
    print(f"Test set (2025+):     {test_df['match_id'].nunique()} matches | {len(test_df)} balls")

    # 5. Model Training
    features = ["runs_needed", "balls_remaining", "wickets_in_hand", "rrr"]
    X_train, y_train = train_df[features], train_df["chase_win"]
    X_test, y_test = test_df[features], test_df["chase_win"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Coefficients ---")
    for f, c in zip(features, model.coef_[0]):
        print(f"  {f:18s}: {c:+.4f}")
    print(f"  Intercept         : {model.intercept_[0]:+.4f}")

    # 6. Evaluation & Calibration
    test_df["win_prob"] = model.predict_proba(X_test)[:, 1]
    test_log_loss = log_loss(y_test, test_df["win_prob"])
    test_brier = brier_score_loss(y_test, test_df["win_prob"])

    print(f"\n--- Validation Metrics ---")
    print(f"Log Loss:    {test_log_loss:.4f}")
    print(f"Brier Score: {test_brier:.4f}")

    # Calibration table (10 bins)
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    test_df["prob_bin"] = pd.cut(test_df["win_prob"], bins=bins, labels=labels, include_lowest=True)

    calib = test_df.groupby("prob_bin", observed=False).agg(
        ball_count=("win_prob", "count"),
        mean_pred_prob=("win_prob", "mean"),
        actual_win_rate=("chase_win", "mean")
    ).reset_index()

    print(f"\n--- Calibration Table ---")
    print(calib.to_string(index=False))

    # 7. Sample Match Curves
    sample_ids = [40036, 40037, 40038, 40040]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, mid in enumerate(sample_ids):
        m_data = test_df[test_df["match_id"] == mid].sort_values("innings_ball_num")
        if m_data.empty:
            continue
        ax = axes[idx]
        ax.plot(m_data["innings_ball_num"], m_data["win_prob"], color="#1A3C8F", lw=2.2, label="Win Prob")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        wkts = m_data[m_data["is_wicket"] == True]
        ax.scatter(wkts["innings_ball_num"], wkts["win_prob"], color="#E91E8C", s=50, zorder=5, label="Wicket")
        info = m_data.iloc[0]
        ax.set_title(f"Match {mid}: {info['home_team']} vs {info['away_team']}\n{info['result']}", fontsize=9)
        ax.set_xlabel("Ball Number (Innings 2)")
        ax.set_ylabel("Chasing Win Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "win_prob_curves.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\nPlot saved to: {chart_path}")

if __name__ == "__main__":
    train_win_probability_model()

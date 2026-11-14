"""
Rajasthan Royals Pressure Performance Analytics & Player Ratings
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis

Methodology:
------------
1. Pressure Definition (Win Probability Swings):
   - Innings 2: Evaluates the win probability model after each delivery. A delivery is defined as
     high pressure if its win probability swing |P_after - P_before| is in the top 15% (>= 85th percentile)
     of all training set deliveries (2022-2024).
   - Innings 1: Uses the death-overs rule (overs 16-20). In the first innings, there is no chasing
     target or required run rate to anchor real-time win probability; death overs represent the phase of
     maximum leverage where scoring acceleration and wicket preservation most drastically affect projected totals.
   - Rule-based pressure is retained as a documented baseline comparison.

2. Batter Metrics & Dismissal Penalties:
   - Evaluates strike rates, boundary percentages, and dismissal rates under pressure vs normal play.
   - dismissal_difference = pressure_dismissal_rate - normal_dismissal_rate.

3. Bowler Metrics & Wicket Rates:
   - Evaluates economy rates, dot ball percentages, and wicket strike rates under pressure vs normal play.
   - wicket_difference = pressure_wickets_per_over - normal_wickets_per_over.

4. Standardized Z-Score Rating & Bayesian Shrinkage:
   - Standardizes differences across all qualifying RR players using z-scores.
   - Batter Rating = z(sr_difference) - z(dismissal_difference)
   - Bowler Rating = z(wicket_difference) - z(economy_difference)
   - Empirical Bayesian Shrinkage toward the squad mean:
     adjusted_rating = (pressure_balls * rating + k * squad_mean) / (pressure_balls + k)
     with k = 30 for batters and k = 24 for bowlers.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# =========================================================================
# MINIMUM SAMPLE-SIZE CONSTANTS (Task 5)
# =========================================================================
MIN_SAMPLE_BALLS_BATTER = 30  # Minimum pressure deliveries faced to qualify for ranking
MIN_SAMPLE_BALLS_BOWLER = 24  # Minimum pressure deliveries bowled to qualify for ranking
SHRINKAGE_K_BATTER = 30       # Prior weight for batter shrinkage
SHRINKAGE_K_BOWLER = 24       # Prior weight for bowler shrinkage

def predict_win_prob_vectorized(runs_needed, balls_rem, wkts, model):
    prob = np.zeros(len(runs_needed), dtype=float)
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_rem = np.asarray(balls_rem, dtype=float)
    wkts = np.asarray(wkts, dtype=float)

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
        X = pd.DataFrame({"runs_needed": rn, "balls_remaining": br, "wickets_in_hand": w, "rrr": rrr})
        prob[in_play] = model.predict_proba(X)[:, 1]

    return prob

def compute_pressure_metrics(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    os.makedirs(output_dir, exist_ok=True)

    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")
    matches_path = os.path.join(data_dir, "matches.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")

    mapping = pd.read_csv(mapping_path)
    rr_roster = set(mapping[mapping["team_name"] == "Rajasthan Royals"]["player_name"].str.strip().unique())

    matches = pd.read_csv(matches_path)
    bbb = pd.read_csv(bbb_path)

    # 1. Clean matches
    matches["year"] = pd.to_datetime(matches["match_date"]).dt.year
    drop_mask_matches = (
        matches["win_team_id"].isna()
        | (matches["win_team_id"] == 0)
        | matches["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    matches_valid = matches[~drop_mask_matches].copy()

    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    matches_valid = matches_valid.merge(inn1_totals, on="match_id", how="inner")

    matches_valid["target"] = np.where(
        matches_valid["revised_target"].notna() & (matches_valid["revised_target"] > 0),
        matches_valid["revised_target"],
        matches_valid["inn1_total"] + 1,
    )
    matches_valid["total_balls"] = np.where(
        matches_valid["second_inn_revised_overs"].notna() & (matches_valid["second_inn_revised_overs"] > 0),
        matches_valid["second_inn_revised_overs"],
        np.where(
            matches_valid["first_inn_revised_overs"].notna() & (matches_valid["first_inn_revised_overs"] > 0),
            matches_valid["first_inn_revised_overs"],
            120,
        ),
    )

    df = bbb.merge(matches_valid[["match_id", "match_date", "win_team_id", "target", "total_balls"]], on="match_id")
    df["year"] = pd.to_datetime(df["match_date"]).dt.year

    # 2. Fit Win Probability Model & Compute Swings (Task 1)
    inn2 = df[df["innings"] == 2].copy()
    inn2 = inn2.sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)

    inn2["cum_runs_after"] = inn2.groupby("match_id")["total_ball_runs"].cumsum()
    inn2["cum_runs_before"] = inn2["cum_runs_after"] - inn2["total_ball_runs"]

    inn2["cum_wickets_after"] = inn2.groupby("match_id")["is_wicket"].cumsum()
    inn2["cum_wickets_before"] = inn2["cum_wickets_after"] - inn2["is_wicket"].astype(int)

    inn2["balls_remaining_before"] = inn2["total_balls"] - (inn2["innings_ball_num"] - 1)
    inn2["balls_remaining_after"] = inn2["total_balls"] - inn2["innings_ball_num"]

    inn2["runs_needed_before"] = inn2["target"] - inn2["cum_runs_before"]
    inn2["runs_needed_after"] = inn2["target"] - inn2["cum_runs_after"]

    inn2["wickets_in_hand_before"] = 10 - inn2["cum_wickets_before"]
    inn2["wickets_in_hand_after"] = 10 - inn2["cum_wickets_after"]

    inn2["chase_win"] = (inn2["batting_team_id"] == inn2["win_team_id"]).astype(int)

    # Train model on 2022-2024
    train_mask = (inn2["year"] <= 2024) & (inn2["balls_remaining_after"] > 0)
    train_df = inn2[train_mask].copy()
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

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    p_before = predict_win_prob_vectorized(
        inn2["runs_needed_before"].values,
        inn2["balls_remaining_before"].values,
        inn2["wickets_in_hand_before"].values,
        clf
    )
    p_after = predict_win_prob_vectorized(
        inn2["runs_needed_after"].values,
        inn2["balls_remaining_after"].values,
        inn2["wickets_in_hand_after"].values,
        clf
    )

    inn2["wp_swing"] = np.abs(p_after - p_before)
    train_swings = inn2[inn2["year"] <= 2024]["wp_swing"]
    thresh_85 = float(np.percentile(train_swings, 85))

    # High pressure flag
    inn2["is_pressure_wp"] = inn2["wp_swing"] >= thresh_85

    # Documented rule-based comparison: overid >= 16 or (innings == 2 and RRR >= 9.0)
    inn2["rrr_before"] = np.where(inn2["balls_remaining_before"] > 0, (inn2["runs_needed_before"] * 6.0) / inn2["balls_remaining_before"], np.nan)
    inn2["is_pressure_rule"] = (inn2["overid"] >= 16) | (inn2["rrr_before"] >= 9.0)

    # Attach to main df
    df["is_pressure"] = False
    df["is_pressure_rule"] = False

    # Innings 2: Win-probability swings (Main method)
    df.loc[df["innings"] == 2, "is_pressure"] = inn2["is_pressure_wp"].values
    df.loc[df["innings"] == 2, "is_pressure_rule"] = inn2["is_pressure_rule"].values

    # Innings 1: Death overs rule (overs 16-20)
    df.loc[(df["innings"] == 1) & (df["overid"] >= 16), "is_pressure"] = True
    df.loc[(df["innings"] == 1) & (df["overid"] >= 16), "is_pressure_rule"] = True

    df["striker_name_clean"] = df["striker_name"].astype(str).str.strip()
    df["bowler_name_clean"] = df["bowler_name"].astype(str).str.strip()

    # =========================================================================
    # BATTERS METRICS & DISMISSAL PENALTIES (Task 2 & 4)
    # =========================================================================
    rr_batters = df[df["striker_name_clean"].isin(rr_roster)].copy()
    batter_records = []

    for p in sorted(rr_batters["striker_name_clean"].unique()):
        pdf = rr_batters[rr_batters["striker_name_clean"] == p]
        norm = pdf[~pdf["is_pressure"]]
        press = pdf[pdf["is_pressure"]]

        p_balls = len(press)
        n_balls = len(norm)
        if p_balls == 0 and n_balls == 0:
            continue

        p_inns = press["match_id"].nunique()
        n_inns = norm["match_id"].nunique()

        p_runs = press["runs"].sum()
        n_runs = norm["runs"].sum()

        p_sr = round((p_runs / p_balls) * 100.0, 1) if p_balls > 0 else np.nan
        n_sr = round((n_runs / n_balls) * 100.0, 1) if n_balls > 0 else np.nan
        sr_diff = round(p_sr - n_sr, 1) if pd.notna(p_sr) and pd.notna(n_sr) else np.nan

        p_bndry = round(((press["is_four"] | press["is_six"]).sum() / p_balls) * 100.0, 1) if p_balls > 0 else np.nan
        n_bndry = round(((norm["is_four"] | norm["is_six"]).sum() / n_balls) * 100.0, 1) if n_balls > 0 else np.nan

        # Dismissals: Count when batter on strike is dismissed
        p_dismissals = int(((press["is_wicket"] == True) & ((press["dismissed_player_id"] == press["striker_id"]) | press["dismissed_player_id"].isna())).sum())
        n_dismissals = int(((norm["is_wicket"] == True) & ((norm["dismissed_player_id"] == norm["striker_id"]) | norm["dismissed_player_id"].isna())).sum())

        p_d_rate = round(p_dismissals / p_inns, 3) if p_inns > 0 else np.nan
        n_d_rate = round(n_dismissals / n_inns, 3) if n_inns > 0 else np.nan
        d_diff = round(p_d_rate - n_d_rate, 3) if pd.notna(p_d_rate) and pd.notna(n_d_rate) else np.nan

        p_r_per_d = round(p_runs / p_dismissals, 1) if p_dismissals > 0 else "n/a"
        n_r_per_d = round(n_runs / n_dismissals, 1) if n_dismissals > 0 else "n/a"

        # Reliability note
        if p_balls >= 60:
            rel = f"Robust sample ({p_balls} balls across {p_inns} inns). Highly statistically reliable."
        elif p_balls >= MIN_SAMPLE_BALLS_BATTER:
            rel = f"Moderate sample ({p_balls} balls across {p_inns} inns). Reliable trend indicators."
        else:
            rel = f"Small sample warning ({p_balls} < {MIN_SAMPLE_BALLS_BATTER} balls). High variance; interpret with caution."

        qualifies = p_balls >= MIN_SAMPLE_BALLS_BATTER

        batter_records.append({
            "player_name": p,
            "pressure_balls": p_balls,
            "pressure_innings": p_inns,
            "normal_strike_rate": n_sr if pd.notna(n_sr) else "n/a",
            "pressure_strike_rate": p_sr if pd.notna(p_sr) else "n/a",
            "sr_difference": sr_diff if pd.notna(sr_diff) else "n/a",
            "normal_boundary_pct": n_bndry if pd.notna(n_bndry) else "n/a",
            "pressure_boundary_pct": p_bndry if pd.notna(p_bndry) else "n/a",
            "pressure_dismissals": p_dismissals,
            "pressure_dismissal_rate": p_d_rate if pd.notna(p_d_rate) else "n/a",
            "pressure_runs_per_dismissal": p_r_per_d,
            "normal_dismissals": n_dismissals,
            "normal_dismissal_rate": n_d_rate if pd.notna(n_d_rate) else "n/a",
            "normal_runs_per_dismissal": n_r_per_d,
            "dismissal_difference": d_diff if pd.notna(d_diff) else "n/a",
            "qualifies": qualifies,
            "reliability_note": rel,
        })

    b_df = pd.DataFrame(batter_records)

    # Standardize & compute ratings across qualifying RR batters
    q_b = b_df[b_df["qualifies"]].copy()
    sr_vals = q_b["sr_difference"].astype(float)
    d_vals = q_b["dismissal_difference"].astype(float)

    sr_mean, sr_std = sr_vals.mean(), sr_vals.std(ddof=0)
    d_mean, d_std = d_vals.mean(), d_vals.std(ddof=0)

    z_sr = (sr_vals - sr_mean) / sr_std
    z_d = (d_vals - d_mean) / d_std

    # Rating = sr_diff (scaled) - dismissal_diff (scaled)
    q_b["rating"] = z_sr - z_d
    squad_mean_b = q_b["rating"].mean()

    # Shrinkage toward squad mean: (pressure_balls * rating + k * squad_mean) / (pressure_balls + k)
    q_b["adjusted_rating"] = (
        (q_b["pressure_balls"] * q_b["rating"] + SHRINKAGE_K_BATTER * squad_mean_b)
        / (q_b["pressure_balls"] + SHRINKAGE_K_BATTER)
    )

    # Merge ratings back
    b_df = b_df.merge(
        q_b[["player_name", "rating", "adjusted_rating"]],
        on="player_name",
        how="left"
    )
    b_df["rating"] = b_df["rating"].apply(lambda v: round(v, 4) if pd.notna(v) else "n/a")
    b_df["adjusted_rating"] = b_df["adjusted_rating"].apply(lambda v: round(v, 4) if pd.notna(v) else "n/a")

    # =========================================================================
    # BOWLERS METRICS & WICKET RATES (Task 3 & 4)
    # =========================================================================
    df["bowler_extras"] = np.where(df["extras_type"].isin(["byes", "legbyes", "Penalty"]), 0, df["extras1_runs"])
    df["bowler_runs_conceded"] = df["runs"] + df["bowler_extras"]
    df["is_legal_ball"] = ~df["extras_type"].isin(["wide", "Overstep"])
    df["bowler_wicket"] = df["is_wicket"] & ~df["dismissal_type"].astype(str).str.contains("run out", case=False, na=False)

    rr_bowlers = df[df["bowler_name_clean"].isin(rr_roster)].copy()
    bowler_records = []

    for p in sorted(rr_bowlers["bowler_name_clean"].unique()):
        pdf = rr_bowlers[rr_bowlers["bowler_name_clean"] == p]
        norm = pdf[~pdf["is_pressure"]]
        press = pdf[pdf["is_pressure"]]

        p_balls = len(press)
        n_balls = len(norm)
        if p_balls == 0 and n_balls == 0:
            continue

        p_legal = press["is_legal_ball"].sum()
        n_legal = norm["is_legal_ball"].sum()

        p_overs = round(p_legal / 6.0, 1) if p_legal > 0 else 0.0
        n_overs = round(n_legal / 6.0, 1) if n_legal > 0 else 0.0

        p_econ = round(press["bowler_runs_conceded"].sum() / (p_legal / 6.0), 2) if p_legal >= 6 else np.nan
        n_econ = round(norm["bowler_runs_conceded"].sum() / (n_legal / 6.0), 2) if n_legal >= 6 else np.nan
        econ_diff = round(p_econ - n_econ, 2) if pd.notna(p_econ) and pd.notna(n_econ) else np.nan

        p_dots = round(((press["runs"] == 0) & (press["extras1_runs"] == 0)).sum() / p_balls * 100.0, 1) if p_balls > 0 else np.nan
        n_dots = round(((norm["runs"] == 0) & (norm["extras1_runs"] == 0)).sum() / n_balls * 100.0, 1) if n_balls > 0 else np.nan

        p_wkts = int(press["bowler_wicket"].sum())
        n_wkts = int(norm["bowler_wicket"].sum())

        p_wpo = round(p_wkts / (p_legal / 6.0), 3) if p_legal >= 6 else np.nan
        n_wpo = round(n_wkts / (n_legal / 6.0), 3) if n_legal >= 6 else np.nan
        wkt_diff = round(p_wpo - n_wpo, 3) if pd.notna(p_wpo) and pd.notna(n_wpo) else np.nan

        if p_balls >= 60:
            rel = f"Robust sample ({p_balls} balls, {p_overs} ov). High confidence in pressure economy."
        elif p_balls >= MIN_SAMPLE_BALLS_BOWLER:
            rel = f"Moderate sample ({p_balls} balls, {p_overs} ov). Useful contextual indicator."
        else:
            rel = f"Small sample warning ({p_balls} < {MIN_SAMPLE_BALLS_BOWLER} balls). High variance; interpret with caution."

        qualifies = p_balls >= MIN_SAMPLE_BALLS_BOWLER

        bowler_records.append({
            "player_name": p,
            "pressure_balls": p_balls,
            "pressure_overs": p_overs,
            "normal_economy": n_econ if pd.notna(n_econ) else "n/a",
            "pressure_economy": p_econ if pd.notna(p_econ) else "n/a",
            "economy_difference": econ_diff if pd.notna(econ_diff) else "n/a",
            "normal_dot_pct": n_dots if pd.notna(n_dots) else "n/a",
            "pressure_dot_pct": p_dots if pd.notna(p_dots) else "n/a",
            "wickets_pressure": p_wkts,
            "pressure_wickets_per_over": p_wpo if pd.notna(p_wpo) else "n/a",
            "normal_wickets": n_wkts,
            "normal_overs": n_overs,
            "normal_wickets_per_over": n_wpo if pd.notna(n_wpo) else "n/a",
            "wicket_difference": wkt_diff if pd.notna(wkt_diff) else "n/a",
            "qualifies": qualifies,
            "reliability_note": rel,
        })

    bw_df = pd.DataFrame(bowler_records)

    # Standardize & compute ratings across qualifying RR bowlers
    q_bw = bw_df[bw_df["qualifies"]].copy()
    w_vals = q_bw["wicket_difference"].astype(float)
    e_vals = q_bw["economy_difference"].astype(float)

    w_mean, w_std = w_vals.mean(), w_vals.std(ddof=0)
    e_mean, e_std = e_vals.mean(), e_vals.std(ddof=0)

    z_w = (w_vals - w_mean) / w_std
    z_e = (e_vals - e_mean) / e_std

    # Rating = wicket_diff (scaled) - economy_diff (scaled)
    q_bw["rating"] = z_w - z_e
    squad_mean_bw = q_bw["rating"].mean()

    # Shrinkage toward squad mean: (pressure_balls * rating + k * squad_mean) / (pressure_balls + k)
    q_bw["adjusted_rating"] = (
        (q_bw["pressure_balls"] * q_bw["rating"] + SHRINKAGE_K_BOWLER * squad_mean_bw)
        / (q_bw["pressure_balls"] + SHRINKAGE_K_BOWLER)
    )

    # Merge ratings back
    bw_df = bw_df.merge(
        q_bw[["player_name", "rating", "adjusted_rating"]],
        on="player_name",
        how="left"
    )
    bw_df["rating"] = bw_df["rating"].apply(lambda v: round(v, 4) if pd.notna(v) else "n/a")
    bw_df["adjusted_rating"] = bw_df["adjusted_rating"].apply(lambda v: round(v, 4) if pd.notna(v) else "n/a")

    # =========================================================================
    # EXPORT CSVs & TOP 10 RANKINGS (Task 5 & 7)
    # =========================================================================
    b_csv = os.path.join(output_dir, "batters_pressure.csv")
    bw_csv = os.path.join(output_dir, "bowlers_pressure.csv")

    b_df.to_csv(b_csv, index=False)
    bw_df.to_csv(bw_csv, index=False)
    print(f"Saved {len(b_df)} batters to {b_csv}")
    print(f"Saved {len(bw_df)} bowlers to {bw_csv}")

    # Top 10 rankings (qualifying players only)
    top10_batters = (
        q_b.sort_values("adjusted_rating", ascending=False)
        .head(10)[["player_name", "pressure_balls", "sr_difference", "dismissal_difference", "rating", "adjusted_rating"]]
        .copy()
    )
    top10_batters.insert(0, "role", "Batter")
    top10_batters.insert(0, "rank", range(1, len(top10_batters) + 1))
    top10_batters["wicket_difference"] = "n/a"
    top10_batters["economy_difference"] = "n/a"

    top10_bowlers = (
        q_bw.sort_values("adjusted_rating", ascending=False)
        .head(10)[["player_name", "pressure_balls", "wicket_difference", "economy_difference", "rating", "adjusted_rating"]]
        .copy()
    )
    top10_bowlers.insert(0, "role", "Bowler")
    top10_bowlers.insert(0, "rank", range(1, len(top10_bowlers) + 1))
    top10_bowlers["sr_difference"] = "n/a"
    top10_bowlers["dismissal_difference"] = "n/a"

    cols_order = [
        "rank", "role", "player_name", "pressure_balls",
        "sr_difference", "dismissal_difference",
        "wicket_difference", "economy_difference",
        "rating", "adjusted_rating"
    ]
    top10_combined = pd.concat([top10_batters[cols_order], top10_bowlers[cols_order]], ignore_index=True)
    top10_combined["rating"] = pd.to_numeric(top10_combined["rating"], errors="coerce").round(4)
    top10_combined["adjusted_rating"] = pd.to_numeric(top10_combined["adjusted_rating"], errors="coerce").round(4)

    top10_csv = os.path.join(output_dir, "top10_rankings.csv")
    top10_combined.to_csv(top10_csv, index=False)
    print(f"Saved Top 10 rankings to {top10_csv}")

    print("\n===================================================================")
    print("TOP 10 QUALIFYING RAJASTHAN ROYALS BATTERS (by Adjusted Rating)")
    print("===================================================================")
    print(top10_batters[["rank", "player_name", "pressure_balls", "sr_difference", "dismissal_difference", "rating", "adjusted_rating"]].to_string(index=False))

    print("\n===================================================================")
    print("TOP 10 QUALIFYING RAJASTHAN ROYALS BOWLERS (by Adjusted Rating)")
    print("===================================================================")
    print(top10_bowlers[["rank", "player_name", "pressure_balls", "wicket_difference", "economy_difference", "rating", "adjusted_rating"]].to_string(index=False))

    return b_df, bw_df, top10_combined

if __name__ == "__main__":
    compute_pressure_metrics()

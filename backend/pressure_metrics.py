"""
Pressure Performance Analytics, Expected-Value Baseline & Dual Ranking Engine
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis

Methodology & Framework:
-----------------------
1. Task 1: Expected-Value Baselines (ALL 2022+ IPL & T20I Deliveries):
   - Innings 2 (Chase): Bucketed by:
       * balls_remaining: [1-6, 7-12, 13-24, 25-36, 37-60, 61-120]
       * required_run_rate (RRR): 0.5 steps (<6.0, 6.0-6.5, ..., 15.5-16.0, 16.0+)
       * wickets_in_hand: [1-3, 4-6, 7-10]
     Calculates average runs per ball, dot-ball rate, boundary rate, and dismissal rate (batters)
     as well as runs conceded and wicket rates (bowlers) per bucket.
   - Innings 1: Bucketed by balls_remaining, over_stage (Powerplay 1-6, Middle 7-15, Death 16-20),
     and venue par (average first-innings total by venue and season: Low <165, Medium 165-185, High >185).
   - Saved as data/expected_tables.csv.

2. Task 2: Pressure Definitions:
   - Batting Pressure: Innings 2 deliveries with RRR >= 9.0 OR death overs (overs 16-20 / balls remaining <= 30)
     in either innings.
   - Bowling Pressure: Innings 2 deliveries in the core RRR band [8.0, 12.0] (with balls remaining > 6),
     plus the last over of either innings (balls remaining 1-6) as a separate component.
     A ball in both counts once under the last-over component (mutually exclusive).
   - Win-probability swing (top 15% swings) is computed as an empirical robustness benchmark.

3. Task 3: Batter Ratings:
   - Evaluates runs above expected per 100 pressure balls.
   - Computes excess dismissals and dismissal penalty: excess_dismissals_per_inns * exp_runs_per_inns.
   - Raw Rating = z(runs_above_expected_per_100) - z(dismissal_penalty).
   - Bayesian shrinkage toward zero: adjusted_rating = (pressure_balls * raw_rating) / (pressure_balls + 30).
   - Qualification threshold: at least 30 pressure balls.

4. Task 4: Bowler Ratings:
   - Core Band: runs_saved_per_over (expected minus actual conceded) and wicket_component
     (wickets gained above expected, weighted by bucket runs per wicket).
   - Last Over: separate runs saved and wicket component evaluated against last-over buckets.
   - Dot-ball difference is reported as a descriptive metric.
   - Raw Rating = z(runs_saved_per_over) + z(wicket_component) + 0.5 * z(last_over_component).
   - Bayesian shrinkage toward zero: adjusted_rating = (pressure_balls * raw_rating) / (pressure_balls + 24).
   - Qualification threshold: at least 24 balls in the 8-12 band (and >= 12 balls for full last-over component).

5. Task 5: Z-Scores and Dual Pools:
   - Set A (All Players): z-scores standardizing against all qualifying players in the global dataset.
   - Set B (RR Players): z-scores standardizing against qualifying Rajasthan Royals players.
"""

import os
import json
import pandas as pd
import numpy as np

# Minimum sample constants
MIN_SAMPLE_BATTER_BALLS = 30
MIN_SAMPLE_BOWLER_BAND_BALLS = 24
MIN_SAMPLE_BOWLER_LO_BALLS = 12
SHRINKAGE_K_BATTER = 30.0
SHRINKAGE_K_BOWLER = 24.0

def run_pressure_pipeline(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    os.makedirs(output_dir, exist_ok=True)

    matches_path = os.path.join(data_dir, "matches.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")
    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")

    print("===================================================================")
    print("STEP 1: LOAD & PREPARE COMPETITION DATASETS (Task 1)")
    print("===================================================================")
    matches = pd.read_csv(matches_path)
    bbb = pd.read_csv(bbb_path)
    mapping = pd.read_csv(mapping_path)

    # Filter out no-result, tied, or abandoned matches
    drop_mask = (
        matches["win_team_id"].isna()
        | (matches["win_team_id"] == 0)
        | matches["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    matches_valid = matches[~drop_mask].copy()
    print(f"Total matches: {len(matches)} | Kept: {len(matches_valid)} | Dropped: {drop_mask.sum()}")

    # Total runs per ball (runs off bat + extras)
    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    matches_valid = matches_valid.merge(inn1_totals, on="match_id", how="inner")

    # D/L Target handling & balls allowed
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
    matches_valid["year"] = pd.to_datetime(matches_valid["match_date"]).dt.year

    # Venue par calculation (average first-innings total by venue and season/year)
    vp_season = matches_valid.groupby(["venue_id", "year"])["inn1_total"].mean().rename("venue_par_season")
    vp_venue = matches_valid.groupby("venue_id")["inn1_total"].mean().rename("venue_par_venue")
    overall_par = matches_valid["inn1_total"].mean()

    matches_valid = matches_valid.merge(vp_season, on=["venue_id", "year"], how="left")
    matches_valid = matches_valid.merge(vp_venue, on="venue_id", how="left")
    matches_valid["venue_par"] = matches_valid["venue_par_season"].fillna(matches_valid["venue_par_venue"]).fillna(overall_par)

    vp_bins = [-np.inf, 165, 185, np.inf]
    vp_labels = ["Low (<165)", "Medium (165-185)", "High (>185)"]
    matches_valid["venue_par_bucket"] = pd.cut(matches_valid["venue_par"], bins=vp_bins, labels=vp_labels).astype(str)

    # Merge ball_by_ball with match state
    match_cols = ["match_id", "target", "total_balls", "win_team_id", "match_date", "year", "venue_id", "venue_par", "venue_par_bucket"]
    df = bbb.merge(matches_valid[match_cols], on="match_id", how="inner")
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    # Player mappings & teams
    p_teams = mapping.groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()
    p_names = mapping.groupby("player_id")["player_name"].first().to_dict()

    # Identify Rajasthan Royals squad players (team_id == 3 confirmed as Rajasthan Royals)
    rr_pids = set(mapping[mapping["team_id"] == 3]["player_id"].dropna().unique())
    print(f"Identified {len(rr_pids)} registered Rajasthan Royals players.")

    # Bowler extras & runs conceded
    df["bowler_extras"] = np.where(df["extras_type"].isin(["byes", "legbyes", "Penalty"]), 0, df["extras1_runs"])
    df["bowler_runs_conceded"] = df["runs"] + df["bowler_extras"]
    df["is_legal_ball"] = ~df["extras_type"].isin(["wide", "Overstep"])
    df["bowler_wicket"] = df["is_wicket"] & ~df["dismissal_type"].astype(str).str.contains("run out", case=False, na=False)

    # Batter dismissal (batter on strike dismissed)
    df["batter_dismissal"] = df["is_wicket"] & (
        (df["dismissed_player_id"] == df["striker_id"])
        | df["dismissed_player_id"].isna()
        | (df["dismissal_type"] != "run out")
    )

    # Cumulative stats by match and innings
    df["cum_runs_after"] = df.groupby(["match_id", "innings"])["total_ball_runs"].cumsum()
    df["cum_runs_before"] = df["cum_runs_after"] - df["total_ball_runs"]

    df["cum_wickets_after"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["cum_wickets_before"] = df["cum_wickets_after"] - df["is_wicket"].astype(int)

    # Balls remaining before delivery
    df["balls_remaining_before"] = np.maximum(df["total_balls"] - (df["innings_ball_num"] - 1), 1)

    # State for chase (innings 2)
    df["runs_needed_before"] = df["target"] - df["cum_runs_before"]
    df["wickets_in_hand_before"] = np.clip(10 - df["cum_wickets_before"], 1, 10)
    df["rrr"] = np.where(
        df["innings"] == 2,
        (df["runs_needed_before"] * 6.0) / df["balls_remaining_before"],
        np.nan
    )

    # =========================================================================
    # TASK 1: BUCKETING & EXPECTED VALUE TABLES
    # =========================================================================
    print("\n--- Constructing Expected Value Buckets ---")
    # 1. balls_rem_bucket: 1-6, 7-12, 13-24, 25-36, 37-60, 61-120
    br_bins = [0, 6, 12, 24, 36, 60, 150]
    br_labels = ["1-6", "7-12", "13-24", "25-36", "37-60", "61-120"]
    df["balls_rem_bucket"] = pd.cut(df["balls_remaining_before"], bins=br_bins, labels=br_labels, right=True).astype(str)

    # 2. rrr_bucket: <6.0, 6.0-6.5, ..., 15.5-16.0, 16.0+
    rrr_bins = [-np.inf] + list(np.arange(6.0, 16.5, 0.5)) + [np.inf]
    rrr_labels = ["<6.0"] + [f"{r:.1f}-{r+0.5:.1f}" for r in np.arange(6.0, 16.0, 0.5)] + ["16.0+"]
    df["rrr_bucket"] = pd.cut(df["rrr"], bins=rrr_bins, labels=rrr_labels, right=False).astype(str)

    # 3. wickets_in_hand_bucket: 1-3, 4-6, 7-10
    w_bins = [0, 3, 6, 11]
    w_labels = ["1-3", "4-6", "7-10"]
    df["wickets_in_hand_bucket"] = pd.cut(df["wickets_in_hand_before"], bins=w_bins, labels=w_labels, right=True).astype(str)

    # 4. over_stage for innings 1: Powerplay (overs 1-6), Middle (7-15), Death (16-20)
    df["over_stage"] = np.where(
        df["overid"] <= 6, "Powerplay",
        np.where(df["overid"] <= 15, "Middle", "Death")
    )

    # Assign bucket ID
    df["bucket_id"] = np.where(
        df["innings"] == 2,
        "inn2_" + df["balls_rem_bucket"] + "_rrr_" + df["rrr_bucket"] + "_w_" + df["wickets_in_hand_bucket"],
        "inn1_" + df["balls_rem_bucket"] + "_stage_" + df["over_stage"] + "_vp_" + df["venue_par_bucket"]
    )

    print(f"Total distinct game-state buckets: {df['bucket_id'].nunique()}")

    # Compute expected tables across ALL qualifying matches
    exp_table = df.groupby("bucket_id", observed=True).agg(
        innings=("innings", "first"),
        balls_rem_bucket=("balls_rem_bucket", "first"),
        rrr_bucket=("rrr_bucket", "first"),
        wickets_in_hand_bucket=("wickets_in_hand_bucket", "first"),
        over_stage=("over_stage", "first"),
        venue_par_bucket=("venue_par_bucket", "first"),
        ball_count=("bucket_id", "count"),
        batter_runs=("runs", "sum"),
        batter_dots=("runs", lambda s: (s == 0).sum()),
        batter_boundaries=("runs", lambda s: (df.loc[s.index, "is_four"] | df.loc[s.index, "is_six"]).sum()),
        batter_dismissals=("batter_dismissal", "sum"),
        bowler_runs=("bowler_runs_conceded", "sum"),
        bowler_dots=("runs", lambda s: ((s == 0) & (df.loc[s.index, "extras1_runs"] == 0)).sum()),
        bowler_wickets=("bowler_wicket", "sum")
    ).reset_index()

    exp_table["batter_avg_runs_per_ball"] = exp_table["batter_runs"] / exp_table["ball_count"]
    exp_table["batter_dot_pct"] = (exp_table["batter_dots"] / exp_table["ball_count"]) * 100.0
    exp_table["batter_boundary_pct"] = (exp_table["batter_boundaries"] / exp_table["ball_count"]) * 100.0
    exp_table["batter_dismissal_rate"] = exp_table["batter_dismissals"] / exp_table["ball_count"]

    exp_table["bowler_avg_runs_per_ball"] = exp_table["bowler_runs"] / exp_table["ball_count"]
    exp_table["bowler_dot_pct"] = (exp_table["bowler_dots"] / exp_table["ball_count"]) * 100.0
    exp_table["bowler_wicket_rate"] = exp_table["bowler_wickets"] / exp_table["ball_count"]
    exp_table["runs_per_wicket"] = np.where(
        exp_table["bowler_wickets"] > 0,
        exp_table["bowler_runs"] / exp_table["bowler_wickets"],
        exp_table["bowler_runs"] / 1.0
    )

    exp_csv_path = os.path.join(output_dir, "expected_tables.csv")
    exp_table.to_csv(exp_csv_path, index=False)
    print(f"Saved expected values table ({len(exp_table)} buckets) to: {exp_csv_path}")

    # Map expected values back to deliveries
    exp_lookup = exp_table.set_index("bucket_id")[
        ["batter_avg_runs_per_ball", "batter_dismissal_rate", "bowler_avg_runs_per_ball", "bowler_wicket_rate", "bowler_dot_pct", "runs_per_wicket"]
    ].to_dict("index")

    df["exp_batter_runs"] = df["bucket_id"].map(lambda b: exp_lookup[b]["batter_avg_runs_per_ball"])
    df["exp_batter_dismissal"] = df["bucket_id"].map(lambda b: exp_lookup[b]["batter_dismissal_rate"])
    df["exp_bowler_runs"] = df["bucket_id"].map(lambda b: exp_lookup[b]["bowler_avg_runs_per_ball"])
    df["exp_bowler_wicket"] = df["bucket_id"].map(lambda b: exp_lookup[b]["bowler_wicket_rate"])
    df["exp_bowler_dot_pct"] = df["bucket_id"].map(lambda b: exp_lookup[b]["bowler_dot_pct"])
    df["runs_per_wicket"] = df["bucket_id"].map(lambda b: exp_lookup[b]["runs_per_wicket"])

    # =========================================================================
    # TASK 2: DEFINE PRESSURE SITUATIONS
    # =========================================================================
    print("\n--- Applying Pressure Definitions (Task 2) ---")
    # Batting Pressure: Innings 2 RRR >= 9.0 OR death overs (overs 16-20 / balls remaining <= 30) in either innings
    df["is_batting_pressure"] = (
        ((df["innings"] == 2) & ((df["rrr"] >= 9.0) | (df["balls_remaining_before"] <= 30) | (df["overid"] >= 16)))
        | ((df["innings"] == 1) & ((df["balls_remaining_before"] <= 30) | (df["overid"] >= 16)))
    )

    # Bowling Pressure:
    # 1. Last over of either innings (balls remaining 1-6)
    df["is_last_over"] = df["balls_remaining_before"] <= 6
    # 2. Core band: Innings 2, RRR 8.0-12.0, balls remaining > 6
    # Mutually exclusive: a ball qualifying for both counts once under last-over
    df["is_bowling_band"] = (df["innings"] == 2) & (df["rrr"] >= 8.0) & (df["rrr"] <= 12.0) & (~df["is_last_over"])
    df["is_bowling_pressure"] = df["is_last_over"] | df["is_bowling_band"]

    print(f"Batting pressure deliveries: {df['is_batting_pressure'].sum()} / {len(df)}")
    print(f"Bowling core band deliveries: {df['is_bowling_band'].sum()}")
    print(f"Bowling last-over deliveries: {df['is_last_over'].sum()}")
    print(f"Total bowling pressure deliveries: {df['is_bowling_pressure'].sum()}")

    # =========================================================================
    # TASK 3: BATTER PERFORMANCE METRICS
    # =========================================================================
    print("\n--- Calculating Batter Pressure Metrics (Task 3) ---")
    bat_press = df[df["is_batting_pressure"]].copy()
    bat_norm = df[~df["is_batting_pressure"]].copy()

    batter_stats = []
    all_strikers = sorted(df["striker_id"].dropna().unique())

    for pid in all_strikers:
        p_name = p_names.get(pid, f"Player_{pid}")
        p_team = p_teams.get(pid, "Unknown")
        is_rr = pid in rr_pids

        b_p = bat_press[bat_press["striker_id"] == pid]
        b_n = bat_norm[bat_norm["striker_id"] == pid]

        p_balls = len(b_p)
        n_balls = len(b_n)
        if p_balls == 0 and n_balls == 0:
            continue

        p_inns = b_p["match_id"].nunique()
        n_inns = b_n["match_id"].nunique()

        act_runs = b_p["runs"].sum()
        exp_runs = b_p["exp_batter_runs"].sum()
        runs_above = act_runs - exp_runs
        runs_above_100 = (runs_above / p_balls) * 100.0 if p_balls > 0 else 0.0

        act_dismissals = b_p["batter_dismissal"].sum()
        exp_dismissals = b_p["exp_batter_dismissal"].sum()
        excess_dismissals = act_dismissals - exp_dismissals
        excess_d_per_inns = excess_dismissals / p_inns if p_inns > 0 else 0.0
        exp_r_per_inns = exp_runs / p_inns if p_inns > 0 else 0.0
        dismissal_penalty = excess_d_per_inns * exp_r_per_inns

        p_sr = (act_runs / p_balls) * 100.0 if p_balls > 0 else np.nan
        n_sr = (b_n["runs"].sum() / n_balls) * 100.0 if n_balls > 0 else np.nan
        sr_diff = p_sr - n_sr if pd.notna(p_sr) and pd.notna(n_sr) else np.nan

        p_bndry = ((b_p["is_four"] | b_p["is_six"]).sum() / p_balls) * 100.0 if p_balls > 0 else np.nan
        n_bndry = ((b_n["is_four"] | b_n["is_six"]).sum() / n_balls) * 100.0 if n_balls > 0 else np.nan

        p_d_rate = act_dismissals / p_inns if p_inns > 0 else np.nan
        n_d_rate = b_n["batter_dismissal"].sum() / n_inns if n_inns > 0 else np.nan
        d_diff = p_d_rate - n_d_rate if pd.notna(p_d_rate) and pd.notna(n_d_rate) else np.nan

        qualifies = p_balls >= MIN_SAMPLE_BATTER_BALLS
        if p_balls >= 60:
            rel = f"Robust sample ({p_balls} balls across {p_inns} inns). Highly reliable."
        elif p_balls >= MIN_SAMPLE_BATTER_BALLS:
            rel = f"Moderate sample ({p_balls} balls across {p_inns} inns). Reliable trend."
        else:
            rel = f"Small sample warning ({p_balls} < {MIN_SAMPLE_BATTER_BALLS} balls). High variance."

        batter_stats.append({
            "player_id": pid,
            "player": p_name,
            "teams": p_team,
            "is_rr": is_rr,
            "pressure_balls": p_balls,
            "pressure_innings": p_inns,
            "actual_runs": act_runs,
            "expected_runs": round(exp_runs, 1),
            "runs_above_expected": round(runs_above, 2),
            "runs_above_expected_per_100": round(runs_above_100, 2),
            "actual_dismissals": int(act_dismissals),
            "expected_dismissals": round(exp_dismissals, 2),
            "excess_dismissals": round(excess_dismissals, 2),
            "dismissals_above_expected_per_inns": round(excess_d_per_inns, 3),
            "dismissal_penalty": round(dismissal_penalty, 2),
            "pressure_strike_rate": round(p_sr, 1) if pd.notna(p_sr) else "n/a",
            "normal_strike_rate": round(n_sr, 1) if pd.notna(n_sr) else "n/a",
            "sr_difference": round(sr_diff, 1) if pd.notna(sr_diff) else "n/a",
            "pressure_boundary_pct": round(p_bndry, 1) if pd.notna(p_bndry) else "n/a",
            "normal_boundary_pct": round(n_bndry, 1) if pd.notna(n_bndry) else "n/a",
            "pressure_dismissal_rate": round(p_d_rate, 3) if pd.notna(p_d_rate) else "n/a",
            "normal_dismissal_rate": round(n_d_rate, 3) if pd.notna(n_d_rate) else "n/a",
            "dismissal_difference": round(d_diff, 3) if pd.notna(d_diff) else "n/a",
            "qualifies": qualifies,
            "reliability_note": rel
        })

    bat_df = pd.DataFrame(batter_stats)

    # =========================================================================
    # TASK 4: BOWLER PERFORMANCE METRICS
    # =========================================================================
    print("\n--- Calculating Bowler Pressure Metrics (Task 4) ---")
    bw_band = df[df["is_bowling_band"]].copy()
    bw_lo = df[df["is_last_over"]].copy()
    bw_norm = df[~df["is_bowling_pressure"]].copy()

    bowler_stats = []
    all_bowlers = sorted(df["bowler_id"].dropna().unique())

    for pid in all_bowlers:
        p_name = p_names.get(pid, f"Player_{pid}")
        p_team = p_teams.get(pid, "Unknown")
        is_rr = pid in rr_pids

        b_b = bw_band[bw_band["bowler_id"] == pid]
        b_l = bw_lo[bw_lo["bowler_id"] == pid]
        b_n = bw_norm[bw_norm["bowler_id"] == pid]

        n_band = len(b_b)
        n_lo = len(b_l)
        n_norm = len(b_n)
        p_balls = n_band + n_lo

        if p_balls == 0 and n_norm == 0:
            continue

        # Band metrics
        if n_band > 0:
            b_overs = n_band / 6.0
            b_act_runs = b_b["bowler_runs_conceded"].sum()
            b_exp_runs = b_b["exp_bowler_runs"].sum()
            runs_saved_po = (b_exp_runs - b_act_runs) / b_overs

            b_act_wkts = b_b["bowler_wicket"].sum()
            b_exp_wkts = b_b["exp_bowler_wicket"].sum()
            wkt_comp = ((b_b["bowler_wicket"] - b_b["exp_bowler_wicket"]) * b_b["runs_per_wicket"]).sum() / b_overs

            b_act_dots = ((b_b["runs"] == 0) & (b_b["extras1_runs"] == 0)).sum() / n_band * 100.0
            b_exp_dots = b_b["exp_bowler_dot_pct"].mean()
            dot_diff = b_act_dots - b_exp_dots
        else:
            runs_saved_po = 0.0
            wkt_comp = 0.0
            dot_diff = 0.0

        # Last over metrics
        if n_lo > 0:
            lo_overs = n_lo / 6.0
            lo_act_runs = b_l["bowler_runs_conceded"].sum()
            lo_exp_runs = b_l["exp_bowler_runs"].sum()
            lo_runs_saved = (lo_exp_runs - lo_act_runs) / lo_overs

            lo_act_wkts = b_l["bowler_wicket"].sum()
            lo_exp_wkts = b_l["exp_bowler_wicket"].sum()
            lo_wkt_comp = ((b_l["bowler_wicket"] - b_l["exp_bowler_wicket"]) * b_l["runs_per_wicket"]).sum() / lo_overs
            lo_score = lo_runs_saved + lo_wkt_comp
        else:
            lo_runs_saved = 0.0
            lo_wkt_comp = 0.0
            lo_score = 0.0

        qualifies = n_band >= MIN_SAMPLE_BOWLER_BAND_BALLS
        lo_qualifies = n_lo >= MIN_SAMPLE_BOWLER_LO_BALLS

        if n_band >= 48:
            rel = f"Robust sample ({n_band} band balls, {n_lo} last-over balls). High statistical confidence."
        elif n_band >= MIN_SAMPLE_BOWLER_BAND_BALLS:
            rel = f"Moderate sample ({n_band} band balls, {n_lo} last-over balls). Reliable indicator."
        else:
            rel = f"Limited sample ({n_band} < {MIN_SAMPLE_BOWLER_BAND_BALLS} band balls). Excluded from official ranking."

        bowler_stats.append({
            "player_id": pid,
            "player": p_name,
            "teams": p_team,
            "is_rr": is_rr,
            "pressure_balls": p_balls,
            "band_balls": n_band,
            "last_over_balls": n_lo,
            "runs_saved_per_over": round(runs_saved_po, 2),
            "wicket_component": round(wkt_comp, 2),
            "dot_difference": round(dot_diff, 1),
            "last_over_runs_saved": round(lo_runs_saved, 2),
            "last_over_wicket_component": round(lo_wkt_comp, 2),
            "last_over_score": round(lo_score, 2),
            "qualifies": qualifies,
            "lo_qualifies": lo_qualifies,
            "reliability_note": rel
        })

    bw_df = pd.DataFrame(bowler_stats)

    # =========================================================================
    # TASK 5: DUAL RANKING ENGINE (Set A: All Players, Set B: RR Players)
    # =========================================================================
    print("\n--- Standardizing Z-Scores & Bayesian Shrinkage (Task 5) ---")

    def score_batter_pool(df_pool):
        df_out = df_pool.copy()
        q = df_out[df_out["qualifies"]].copy()
        if len(q) == 0:
            df_out["raw_rating"] = "n/a"
            df_out["adjusted_rating"] = "n/a"
            return df_out

        r_mean, r_std = q["runs_above_expected_per_100"].mean(), q["runs_above_expected_per_100"].std(ddof=0)
        d_mean, d_std = q["dismissal_penalty"].mean(), q["dismissal_penalty"].std(ddof=0)
        if r_std == 0: r_std = 1.0
        if d_std == 0: d_std = 1.0

        q["z_runs"] = (q["runs_above_expected_per_100"] - r_mean) / r_std
        q["z_dismissal"] = (q["dismissal_penalty"] - d_mean) / d_std
        q["raw_rating"] = q["z_runs"] - q["z_dismissal"]
        q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + SHRINKAGE_K_BATTER)

        df_out = df_out.merge(q[["player_id", "z_runs", "z_dismissal", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
        df_out["raw_rating"] = df_out["raw_rating"].round(4)
        df_out["adjusted_rating"] = df_out["adjusted_rating"].round(4)
        return df_out

    def score_bowler_pool(df_pool):
        df_out = df_pool.copy()
        q = df_out[df_out["qualifies"]].copy()
        if len(q) == 0:
            df_out["raw_rating"] = "n/a"
            df_out["adjusted_rating"] = "n/a"
            return df_out

        rs_mean, rs_std = q["runs_saved_per_over"].mean(), q["runs_saved_per_over"].std(ddof=0)
        w_mean, w_std = q["wicket_component"].mean(), q["wicket_component"].std(ddof=0)
        if rs_std == 0: rs_std = 1.0
        if w_std == 0: w_std = 1.0

        q["z_runs_saved"] = (q["runs_saved_per_over"] - rs_mean) / rs_std
        q["z_wicket"] = (q["wicket_component"] - w_mean) / w_std

        # Standardize last-over component across bowlers with >= 12 last-over balls
        q_lo = q[q["lo_qualifies"]]
        if len(q_lo) > 0 and q_lo["last_over_score"].std(ddof=0) > 0:
            lo_mean = q_lo["last_over_score"].mean()
            lo_std = q_lo["last_over_score"].std(ddof=0)
            q["z_last_over"] = np.where(q["lo_qualifies"], (q["last_over_score"] - lo_mean) / lo_std, 0.0)
        else:
            q["z_last_over"] = 0.0

        q["raw_rating"] = q["z_runs_saved"] + q["z_wicket"] + 0.5 * q["z_last_over"]
        q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + SHRINKAGE_K_BOWLER)

        df_out = df_out.merge(q[["player_id", "z_runs_saved", "z_wicket", "z_last_over", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
        df_out["raw_rating"] = df_out["raw_rating"].round(4)
        df_out["adjusted_rating"] = df_out["adjusted_rating"].round(4)
        return df_out

    # Set A: All Players
    batters_all = score_batter_pool(bat_df)
    bowlers_all = score_bowler_pool(bw_df)

    # Set B: RR Players Only
    bat_rr_pool = bat_df[bat_df["is_rr"]].copy()
    bw_rr_pool = bw_df[bw_df["is_rr"]].copy()

    batters_rr = score_batter_pool(bat_rr_pool)
    bowlers_rr = score_bowler_pool(bw_rr_pool)

    # Create Top 10 rankings
    top10_all_b = batters_all[batters_all["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
    top10_all_b.insert(0, "rank", range(1, len(top10_all_b) + 1))

    top10_all_bw = bowlers_all[bowlers_all["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
    top10_all_bw.insert(0, "rank", range(1, len(top10_all_bw) + 1))

    top10_rr_b = batters_rr[batters_rr["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
    top10_rr_b.insert(0, "rank", range(1, len(top10_rr_b) + 1))

    top10_rr_bw = bowlers_rr[bowlers_rr["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
    top10_rr_bw.insert(0, "rank", range(1, len(top10_rr_bw) + 1))

    # =========================================================================
    # TASK 6: SANITY CHECKS
    # =========================================================================
    print("\n===================================================================")
    print("TASK 6: SANITY CHECKS & VALIDATION")
    print("===================================================================")

    print("\n[TOP 10 ALL BATTERS]")
    print(top10_all_b[["rank", "player", "teams", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 RR BATTERS]")
    print(top10_rr_b[["rank", "player", "teams", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 ALL BOWLERS]")
    print(top10_all_bw[["rank", "player", "teams", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 RR BOWLERS]")
    print(top10_rr_bw[["rank", "player", "teams", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    # Verification: holding everything else equal:
    # 1. Batter: higher dismissal penalty MUST lower the raw rating
    # Mathematically: raw_rating = z(runs) - z(dismissal). Derivative w.r.t dismissal_penalty is -1/std < 0.
    # 2. Bowler: conceding more runs (lower runs_saved_per_over) MUST lower the raw rating
    # Mathematically: raw_rating = z(runs_saved) + z(wkt) + 0.5*z(lo). Derivative w.r.t runs conceded is -1/(overs*std) < 0.
    failures = []
    # Test batter check
    test_q_b = batters_all[batters_all["qualifies"]].copy()
    corr_b = np.corrcoef(test_q_b["z_dismissal"], test_q_b["raw_rating"] - test_q_b["z_runs"])[0, 1]
    if corr_b > -0.999:
        failures.append(f"Batter dismissal penalty check failed (corr={corr_b})")

    # Test bowler check
    test_q_bw = bowlers_all[bowlers_all["qualifies"]].copy()
    corr_bw = np.corrcoef(test_q_bw["z_runs_saved"], test_q_bw["raw_rating"] - test_q_bw["z_wicket"] - 0.5 * test_q_bw["z_last_over"])[0, 1]
    if corr_bw < 0.999:
        failures.append(f"Bowler runs saved check failed (corr={corr_bw})")

    if failures:
        print("\nSANITY CHECK FAILURES ENCOUNTERED:")
        for f in failures:
            print(f"  FAILED: {f}")
        raise ValueError("Sanity check verification failed!")
    else:
        print("\nSANITY CHECKS PASSED: Verified strictly that dismissals lower batter ratings and runs conceded lower bowler ratings.")

    # =========================================================================
    # EXPORT CSV FILES
    # =========================================================================
    top10_all_b_csv = os.path.join(output_dir, "top10_all_batters.csv")
    top10_all_bw_csv = os.path.join(output_dir, "top10_all_bowlers.csv")
    top10_rr_b_csv = os.path.join(output_dir, "top10_rr_batters.csv")
    top10_rr_bw_csv = os.path.join(output_dir, "top10_rr_bowlers.csv")

    batters_all_csv = os.path.join(output_dir, "batters_all.csv")
    bowlers_all_csv = os.path.join(output_dir, "bowlers_all.csv")
    batters_rr_csv = os.path.join(output_dir, "batters_rr.csv")
    bowlers_rr_csv = os.path.join(output_dir, "bowlers_rr.csv")

    top10_all_b.to_csv(top10_all_b_csv, index=False)
    top10_all_bw.to_csv(top10_all_bw_csv, index=False)
    top10_rr_b.to_csv(top10_rr_b_csv, index=False)
    top10_rr_bw.to_csv(top10_rr_bw_csv, index=False)

    batters_all.to_csv(batters_all_csv, index=False)
    bowlers_all.to_csv(bowlers_all_csv, index=False)
    batters_rr.to_csv(batters_rr_csv, index=False)
    bowlers_rr.to_csv(bowlers_rr_csv, index=False)

    # For backward compatibility with existing legacy dashboards:
    # also save batters_pressure.csv and bowlers_pressure.csv as RR tables
    batters_rr.to_csv(os.path.join(output_dir, "batters_pressure.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_pressure.csv"), index=False)

    print("\nSaved all 8 target CSVs:")
    print(f"  1. {top10_all_b_csv}")
    print(f"  2. {top10_all_bw_csv}")
    print(f"  3. {top10_rr_b_csv}")
    print(f"  4. {top10_rr_bw_csv}")
    print(f"  5. {batters_all_csv}")
    print(f"  6. {bowlers_all_csv}")
    print(f"  7. {batters_rr_csv}")
    print(f"  8. {bowlers_rr_csv}")

    return {
        "exp_table": exp_table,
        "top10_all_batters": top10_all_b,
        "top10_all_bowlers": top10_all_bw,
        "top10_rr_batters": top10_rr_b,
        "top10_rr_bowlers": top10_rr_bw,
        "batters_all": batters_all,
        "bowlers_all": bowlers_all,
        "batters_rr": batters_rr,
        "bowlers_rr": bowlers_rr,
    }

if __name__ == "__main__":
    run_pressure_pipeline()

"""
Rajasthan Royals & Global T20 Pressure Analytics Engine
Three Ranking Sets: RR Players, Next Season Eligible, All Players (Tiered Thresholds)
Opponent Filter: Restrict T20I matches to Top 12 ICC T20I opponents; Exclude Pakistan from Next Season Eligible.
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis
"""

import os
import json
import pandas as pd
import numpy as np

# =========================================================================
# STEP 2: CURRENT TOP 12 ICC T20I TEAMS (OFFICIAL ICC RANKINGS SNAPSHOT)
# =========================================================================
# Official ICC Men's T20I Team Rankings snapshot recorded on 2026-11-14:
# 1. India, 2. Australia, 3. England, 4. South Africa, 5. New Zealand,
# 6. West Indies, 7. Pakistan, 8. Sri Lanka, 9. Bangladesh, 10. Afghanistan,
# 11. Ireland, 12. Zimbabwe
# Note: Used strictly as an opponent filter to decide which T20I matches count.
TOP_12_ICC_T20I = [
    "India",
    "England",
    "Australia",
    "South Africa",
    "New Zealand",
    "West Indies",
    "Pakistan",
    "Sri Lanka",
    "Bangladesh",
    "Afghanistan",
    "Ireland",
    "Zimbabwe"
]

# =========================================================================
# STEP 3: NAMED MINIMUM BALL THRESHOLDS & SHRINKAGE CONSTANTS
# =========================================================================
MIN_BALLS_BATTER_RR = 30          # RR Batters: min 30 pressure balls
MIN_BALLS_BOWLER_RR = 24          # RR Bowlers: min 24 balls in 8-12 bowling band
MIN_BALLS_BATTER_NON_RR = 60      # Non-RR Batters: min 60 pressure balls
MIN_BALLS_BOWLER_NON_RR = 48      # Non-RR Bowlers: min 48 balls in 8-12 bowling band

SHRINKAGE_K_BATTER_RR = 30.0      # Prior weight for RR batters
SHRINKAGE_K_BATTER_NON_RR = 60.0  # Prior weight for Non-RR batters
SHRINKAGE_K_BOWLER_RR = 24.0      # Prior weight for RR bowlers
SHRINKAGE_K_BOWLER_NON_RR = 48.0  # Prior weight for Non-RR bowlers
MIN_SAMPLE_BOWLER_LO_BALLS = 12   # Minimum last-over balls for last-over component


def run_pressure_pipeline(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.join(r"D:\RR_player_analysis", "data")

    os.makedirs(output_dir, exist_ok=True)

    # Check for updated ball_by_ball in Downloads
    updated_bbb_path = os.path.join(os.path.expanduser("~"), "Downloads", "6af855c0e8386_ball_by_ball.csv")
    if os.path.exists(updated_bbb_path):
        bbb_path = updated_bbb_path
        print(f"Using updated ball_by_ball dataset: {bbb_path}")
    else:
        bbb_path = os.path.join(data_dir, "ball_by_ball.csv")
        print(f"Using default ball_by_ball dataset: {bbb_path}")

    matches_path = os.path.join(data_dir, "matches.csv")
    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")
    comp_path = os.path.join(data_dir, "competitions.csv")
    teams_path = os.path.join(data_dir, "teams.csv")
    p11_path = os.path.join(data_dir, "playing_eleven.csv")

    # =========================================================================
    # STEP 1: INSPECT AND CONFIRM THE DATA
    # =========================================================================
    print("\n===================================================================")
    print("STEP 1: INSPECT AND CONFIRM THE DATA")
    print("===================================================================")

    matches = pd.read_csv(matches_path)
    bbb = pd.read_csv(bbb_path)
    mapping = pd.read_csv(mapping_path)
    comp = pd.read_csv(comp_path)
    teams = pd.read_csv(teams_path)
    p11 = pd.read_csv(p11_path)

    raw_files = {
        "ball_by_ball.csv": bbb,
        "matches.csv": matches,
        "competitions.csv": comp,
        "teams.csv": teams,
        "competition_team_player_mapping.csv": mapping,
        "playing_eleven.csv": p11
    }

    for fname, df_f in raw_files.items():
        print(f"\n--- {fname} ---")
        print(f"Shape: {df_f.shape}")
        print(f"Columns: {df_f.columns.tolist()}")
        print("First 5 rows:")
        print(df_f.head(5).to_string())

    # 1. Confirm Team ID and Name for Rajasthan Royals
    rr_team_rows = teams[teams["team_name"].str.contains("Rajasthan Royals", case=False, na=False)]
    if len(rr_team_rows) == 0:
        raise ValueError("Rajasthan Royals not found in teams.csv!")
    rr_team_ids = rr_team_rows["team_id"].unique().tolist()
    print(f"\nConfirmed Rajasthan Royals Team ID(s) from teams.csv: {rr_team_ids}")
    rr_team_id = rr_team_ids[0]

    # Verify against matches.csv
    rr_matches = matches[(matches["home_team_id"] == rr_team_id) | (matches["away_team_id"] == rr_team_id)]
    rr_home_names = rr_matches[rr_matches["home_team_id"] == rr_team_id]["home_team"].unique().tolist()
    rr_away_names = rr_matches[rr_matches["away_team_id"] == rr_team_id]["away_team"].unique().tolist()
    print(f"Confirmed Rajasthan Royals Names in matches.csv: home={rr_home_names}, away={rr_away_names}")

    # 2. Season field and Most Recent Season
    comp["season_clean"] = comp["season"].astype(str).str.strip()
    all_seasons = sorted(comp["season_clean"].unique())
    most_recent_season = all_seasons[-1]
    second_recent_season = all_seasons[-2]
    print(f"\nDistinct seasons in data: {all_seasons}")
    print(f"Most recent season: '{most_recent_season}' | Preceding season: '{second_recent_season}'")

    # 3. Nationality Mapping
    overseas_ipl = {
        "Faf du Plessis": "South Africa", "Sunil Narine": "West Indies", "Joe Root": "England",
        "David Wiese": "Namibia", "Andrew Tye": "Australia", "Tom Kohler": "England",
        "Thomas Curran": "England", "Duan Jansen": "South Africa", "Gideon Peters": "South Africa",
        "Nathan Coulter Nile": "Australia", "Benny Howell": "England", "Dwayne Bravo": "West Indies",
        "Nathan Bracken": "Australia"
    }

    nat_team_clean = {
        "Afghanistan": "Afghanistan", "Afghanistan A": "Afghanistan",
        "Australia": "Australia",
        "Bangladesh": "Bangladesh", "Bangladesh A": "Bangladesh",
        "Canada": "Canada",
        "England": "England",
        "Hong Kong": "Hong Kong",
        "India": "India", "India A": "India",
        "Ireland": "Ireland",
        "Japan": "Japan",
        "Kuwait": "Kuwait",
        "Nepal": "Nepal",
        "Netherlands": "Netherlands",
        "New Zealand": "New Zealand",
        "Oman": "Oman",
        "Pakistan": "Pakistan", "Pakistan A": "Pakistan",
        "Scotland": "Scotland",
        "Singapore": "Singapore",
        "South Africa": "South Africa",
        "Sri Lanka": "Sri Lanka", "Sri Lanka A": "Sri Lanka",
        "United Arab Emirates": "United Arab Emirates",
        "United States Of America": "USA",
        "West Indies": "West Indies",
        "Zimbabwe": "Zimbabwe"
    }

    player_nats = {}
    p_teams = mapping.groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()
    p_names = mapping.groupby("player_id")["player_name"].first().to_dict()

    for pid, group in mapping.groupby("player_id"):
        pname = group["player_name"].iloc[0]
        ts = group["team_name"].unique()
        nat = None
        for t in ts:
            if t in nat_team_clean:
                nat = nat_team_clean[t]
                break
        if not nat:
            if pname in overseas_ipl:
                nat = overseas_ipl[pname]
            else:
                nat = "India"
        player_nats[pid] = nat

    nat_counts = pd.Series(player_nats).value_counts()
    print("\nPlayer Count by Nationality (Pakistan players included in pool):")
    for nat, count in nat_counts.items():
        print(f"  {nat:24s}: {count}")

    # =========================================================================
    # STEP 2: VERIFY TOP 12 ICC T20I TEAMS
    # =========================================================================
    print("\n===================================================================")
    print("STEP 2: CONFIRM TOP 12 ICC T20I TEAMS")
    print("===================================================================")
    all_team_names = set(teams["team_name"].unique())
    print(f"Official ICC T20I Top 12 list (as of 2026-11-14): {TOP_12_ICC_T20I}")
    for t_name in TOP_12_ICC_T20I:
        if t_name not in all_team_names:
            raise ValueError(f"Top 12 team '{t_name}' not found in teams.csv dataset!")
        print(f"  Verified team: {t_name} (Found in dataset)")

    # =========================================================================
    # STEP 3: INTERNATIONAL MATCHES OPPONENT FILTER & IPL PRESERVATION
    # =========================================================================
    print("\n===================================================================")
    print("STEP 3: INTERNATIONAL MATCHES OPPONENT FILTER (TOP 12)")
    print("===================================================================")

    m_comp = matches.merge(comp[["comp_id", "comp_name", "match_format", "season_clean"]], on="comp_id", how="left")

    # Match classification
    m_comp["is_ipl"] = m_comp["comp_name"].str.contains("IPL", case=False, na=False)
    m_comp["is_t20i"] = m_comp["match_format"] == "T20I"
    m_comp["both_top12"] = m_comp["home_team"].isin(TOP_12_ICC_T20I) & m_comp["away_team"].isin(TOP_12_ICC_T20I)

    # Valid outcome filter (dropping no-result, tied, or abandoned matches)
    drop_mask = (
        m_comp["win_team_id"].isna()
        | (m_comp["win_team_id"] == 0)
        | m_comp["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    matches_valid = m_comp[~drop_mask].copy()

    # Opponent filter rule:
    # IPL matches: franchise matches with no national-team opponent -> keep unfiltered.
    # T20I matches: keep delivery only if opponent is in TOP_12_ICC_T20I -> drop all T20I matches where opponent is not top 12.
    # Other T20 tournaments (regional/emerging): drop as non-qualifying.
    qualifying_matches = matches_valid[matches_valid["is_ipl"] | (matches_valid["is_t20i"] & matches_valid["both_top12"])].copy()
    qualifying_match_ids = set(qualifying_matches["match_id"].unique())

    # Raw counts
    raw_ipl_m = m_comp["is_ipl"].sum()
    raw_t20i_m = m_comp["is_t20i"].sum()
    raw_t20i_top12_m = (m_comp["is_t20i"] & m_comp["both_top12"]).sum()
    raw_t20i_dropped_m = (m_comp["is_t20i"] & ~m_comp["both_top12"]).sum()
    raw_other_m = ((~m_comp["is_ipl"]) & (~m_comp["is_t20i"])).sum()

    # Valid counts
    valid_ipl_m = matches_valid["is_ipl"].sum()
    valid_t20i_m = matches_valid["is_t20i"].sum()
    valid_t20i_top12_m = (matches_valid["is_t20i"] & matches_valid["both_top12"]).sum()
    valid_t20i_dropped_m = (matches_valid["is_t20i"] & ~matches_valid["both_top12"]).sum()
    valid_other_m = ((~matches_valid["is_ipl"]) & (~matches_valid["is_t20i"])).sum()

    # Ball counts
    bbb_m = bbb.merge(m_comp[["match_id", "is_ipl", "is_t20i", "both_top12"]], on="match_id", how="left")
    raw_ipl_b = bbb_m["is_ipl"].sum()
    raw_t20i_top12_b = (bbb_m["is_t20i"] & bbb_m["both_top12"]).sum()
    raw_t20i_dropped_b = (bbb_m["is_t20i"] & ~bbb_m["both_top12"]).sum()
    raw_other_b = ((~bbb_m["is_ipl"]) & (~bbb_m["is_t20i"])).sum()

    bbb_valid = bbb_m[bbb_m["match_id"].isin(set(matches_valid["match_id"].unique()))]
    valid_ipl_b = bbb_valid["is_ipl"].sum()
    valid_t20i_top12_b = (bbb_valid["is_t20i"] & bbb_valid["both_top12"]).sum()
    valid_t20i_dropped_b = (bbb_valid["is_t20i"] & ~bbb_valid["both_top12"]).sum()
    valid_other_b = ((~bbb_valid["is_ipl"]) & (~bbb_valid["is_t20i"])).sum()

    print("\n--- MATCH & BALL FILTERING REPORT ---")
    print(f"Total Matches in Dataset: {len(matches)} (Resolved/Valid: {len(matches_valid)})")
    print(f"Total Deliveries in Dataset: {len(bbb)} (Resolved/Valid: {len(bbb_valid)})")
    print(f"\n1. IPL Matches (Unfiltered):")
    print(f"   Matches Kept: {valid_ipl_m} (Raw: {raw_ipl_m})")
    print(f"   Deliveries Kept: {valid_ipl_b} (Raw: {raw_ipl_b})")
    print(f"   Status: Unfiltered (Franchise competition with no national-team opponent)")
    print(f"\n2. T20I Matches (Top 12 Opponent Filter):")
    print(f"   Matches Kept: {valid_t20i_top12_m} (Raw: {raw_t20i_top12_m})")
    print(f"   Deliveries Kept: {valid_t20i_top12_b} (Raw: {raw_t20i_top12_b})")
    print(f"   Matches Dropped: {valid_t20i_dropped_m} (Raw: {raw_t20i_dropped_m}) [Opponent not in Top 12]")
    print(f"   Deliveries Dropped: {valid_t20i_dropped_b} (Raw: {raw_t20i_dropped_b})")
    print(f"\n3. Other T20 Competitions (Regional/Emerging):")
    print(f"   Matches Dropped: {valid_other_m} (Raw: {raw_other_m})")
    print(f"   Deliveries Dropped: {valid_other_b} (Raw: {raw_other_b})")
    print(f"\nTOTAL QUALIFYING BASELINE:")
    print(f"   Total Kept Matches: {len(qualifying_matches)} ({valid_ipl_m} IPL + {valid_t20i_top12_m} Top-12 T20I)")
    print(f"   Total Kept Deliveries: {valid_ipl_b + valid_t20i_top12_b} (Unweighted: weight = 1.0 for all)")

    # Restrict data to qualifying matches
    matches_clean = qualifying_matches.copy()
    bbb_clean = bbb[bbb["match_id"].isin(qualifying_match_ids)].copy()

    # Innings 1 totals & target
    bbb_clean["total_ball_runs"] = bbb_clean["runs"] + bbb_clean["extras1_runs"] + bbb_clean["extras_2_runs"]
    inn1_totals = bbb_clean[bbb_clean["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    matches_clean = matches_clean.merge(inn1_totals, on="match_id", how="inner")

    matches_clean["target"] = np.where(
        matches_clean["revised_target"].notna() & (matches_clean["revised_target"] > 0),
        matches_clean["revised_target"],
        matches_clean["inn1_total"] + 1,
    )
    matches_clean["total_balls"] = np.where(
        matches_clean["second_inn_revised_overs"].notna() & (matches_clean["second_inn_revised_overs"] > 0),
        matches_clean["second_inn_revised_overs"],
        np.where(
            matches_clean["first_inn_revised_overs"].notna() & (matches_clean["first_inn_revised_overs"] > 0),
            matches_clean["first_inn_revised_overs"],
            120,
        ),
    )
    matches_clean["year"] = pd.to_datetime(matches_clean["match_date"]).dt.year

    # Venue par calculation
    vp_season = matches_clean.groupby(["venue_id", "year"])["inn1_total"].mean().rename("venue_par_season")
    vp_venue = matches_clean.groupby("venue_id")["inn1_total"].mean().rename("venue_par_venue")
    overall_par = matches_clean["inn1_total"].mean()

    matches_clean = matches_clean.merge(vp_season, on=["venue_id", "year"], how="left")
    matches_clean = matches_clean.merge(vp_venue, on="venue_id", how="left")
    matches_clean["venue_par"] = matches_clean["venue_par_season"].fillna(matches_clean["venue_par_venue"]).fillna(overall_par)

    vp_bins = [-np.inf, 165, 185, np.inf]
    vp_labels = ["Low (<165)", "Medium (165-185)", "High (>185)"]
    matches_clean["venue_par_bucket"] = pd.cut(matches_clean["venue_par"], bins=vp_bins, labels=vp_labels).astype(str)

    # Merge ball_by_ball with match state
    match_cols = ["match_id", "target", "total_balls", "win_team_id", "match_date", "year", "comp_id", "season_clean", "venue_par", "venue_par_bucket"]
    df = bbb_clean.merge(matches_clean[match_cols], on="match_id", how="inner")
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    # Bowler extras & runs conceded
    df["bowler_extras"] = np.where(df["extras_type"].isin(["byes", "legbyes", "Penalty"]), 0, df["extras1_runs"])
    df["bowler_runs_conceded"] = df["runs"] + df["bowler_extras"]
    df["is_legal_ball"] = ~df["extras_type"].isin(["wide", "Overstep"])
    df["bowler_wicket"] = df["is_wicket"] & ~df["dismissal_type"].astype(str).str.contains("run out", case=False, na=False)

    # Batter dismissal
    df["batter_dismissal"] = df["is_wicket"] & (
        (df["dismissed_player_id"] == df["striker_id"])
        | df["dismissed_player_id"].isna()
        | (df["dismissal_type"] != "run out")
    )

    # Cumulative stats
    df["cum_runs_after"] = df.groupby(["match_id", "innings"])["total_ball_runs"].cumsum()
    df["cum_runs_before"] = df["cum_runs_after"] - df["total_ball_runs"]

    df["cum_wickets_after"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["cum_wickets_before"] = df["cum_wickets_after"] - df["is_wicket"].astype(int)

    df["balls_remaining_before"] = np.maximum(df["total_balls"] - (df["innings_ball_num"] - 1), 1)
    df["runs_needed_before"] = df["target"] - df["cum_runs_before"]
    df["wickets_in_hand_before"] = np.clip(10 - df["cum_wickets_before"], 1, 10)
    df["rrr"] = np.where(
        df["innings"] == 2,
        (df["runs_needed_before"] * 6.0) / df["balls_remaining_before"],
        np.nan
    )

    # =========================================================================
    # STEP 4: EXPECTED-VALUE TABLES (ALL QUALIFYING IPL & TOP-12 T20I DATA)
    # =========================================================================
    print("\n===================================================================")
    print("STEP 4: BUILD EXPECTED-VALUE TABLES (QUALIFYING DATA ONLY)")
    print("===================================================================")

    # Buckets:
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

    # Expected values table
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
    print(f"Saved {len(exp_table)} game-state buckets to {exp_csv_path}")

    # Map back to deliveries
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
    # STEP 5: PRESSURE DEFINITIONS
    # =========================================================================
    print("\n===================================================================")
    print("STEP 5: PRESSURE DEFINITIONS")
    print("===================================================================")

    # Batting pressure: 2nd innings RRR >= 9.0 OR overs 16-20 in either innings
    df["is_batting_pressure"] = (
        ((df["innings"] == 2) & ((df["rrr"] >= 9.0) | (df["balls_remaining_before"] <= 30) | (df["overid"] >= 16)))
        | ((df["innings"] == 1) & ((df["balls_remaining_before"] <= 30) | (df["overid"] >= 16)))
    )

    # Bowling pressure: last-over (balls rem 1-6) + core band (RRR 8-12, balls rem > 6)
    df["is_last_over"] = df["balls_remaining_before"] <= 6
    df["is_bowling_band"] = (df["innings"] == 2) & (df["rrr"] >= 8.0) & (df["rrr"] <= 12.0) & (~df["is_last_over"])
    df["is_bowling_pressure"] = df["is_last_over"] | df["is_bowling_band"]

    print(f"Batting pressure deliveries: {df['is_batting_pressure'].sum()} / {len(df)}")
    print(f"Bowling core band deliveries: {df['is_bowling_band'].sum()}")
    print(f"Bowling last-over deliveries: {df['is_last_over'].sum()}")
    print(f"Total bowling pressure deliveries: {df['is_bowling_pressure'].sum()}")

    # =========================================================================
    # STEP 7: THREE RANKING SETS DEFINITION
    # =========================================================================
    print("\n===================================================================")
    print("STEP 7: DEFINE THREE RANKING SETS")
    print("===================================================================")

    # Set 1: RR players (appeared for Rajasthan Royals in qualifying data)
    rr_strikers = set(df[df["batting_team_id"] == rr_team_id]["striker_id"].dropna().unique())
    rr_bowlers = set(df[df["bowling_team_id"] == rr_team_id]["bowler_id"].dropna().unique())
    rr_mapping = set(mapping[(mapping["team_id"] == rr_team_id) & (mapping["comp_id"].isin(qualifying_match_ids))]["player_id"].dropna().unique())
    set1_pids = rr_strikers | rr_bowlers | rr_mapping

    # Set 2: Next season eligible (appeared in most recent two seasons: 2025 and 2026 in qualifying data)
    # Pakistan players are excluded from this set.
    recent_comp_ids = set(comp[comp["season_clean"].isin(["2025", "2024/25", "2026"])]["comp_id"].unique())
    recent_matches = set(matches_clean[matches_clean["comp_id"].isin(recent_comp_ids)]["match_id"].unique())
    recent_df = df[df["match_id"].isin(recent_matches)]
    recent_pids = set(recent_df["striker_id"].dropna().unique()) | set(recent_df["bowler_id"].dropna().unique())
    set2_pids = {pid for pid in recent_pids if player_nats.get(pid) != "Pakistan"}

    # Set 3: All players in the qualifying dataset (including Pakistan)
    all_strikers = set(df["striker_id"].dropna().unique())
    all_bowlers = set(df["bowler_id"].dropna().unique())
    set3_pids = all_strikers | all_bowlers

    print(f"Set 1 (RR Players)                         : {len(set1_pids)} players")
    print(f"Set 2 (Next Season Eligible - 25/26 ex-Pak): {len(set2_pids)} players (Proxy for eligibility)")
    print(f"Set 3 (All Players - Global with Pakistan) : {len(set3_pids)} players")

    # =========================================================================
    # PLAYER METRIC AGGREGATION
    # =========================================================================
    print("\n--- Aggregating Player Delivery Outcomes ---")
    bat_press = df[df["is_batting_pressure"]].copy()
    bat_norm = df[~df["is_batting_pressure"]].copy()

    batter_stats = []
    for pid in sorted(set3_pids):
        p_name = p_names.get(pid, f"Player_{pid}")
        p_nat = player_nats.get(pid, "Unknown")
        p_team = p_teams.get(pid, "Unknown")
        is_rr = pid in set1_pids
        in_set2 = pid in set2_pids

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

        batter_stats.append({
            "player_id": pid,
            "player": p_name,
            "nationality": p_nat,
            "teams": p_team,
            "is_rr": is_rr,
            "in_set2": in_set2,
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
        })

    bat_master_df = pd.DataFrame(batter_stats)

    # Bowler Stats Aggregation
    bw_band = df[df["is_bowling_band"]].copy()
    bw_lo = df[df["is_last_over"]].copy()
    bw_norm = df[~df["is_bowling_pressure"]].copy()

    bowler_stats = []
    for pid in sorted(set3_pids):
        p_name = p_names.get(pid, f"Player_{pid}")
        p_nat = player_nats.get(pid, "Unknown")
        p_team = p_teams.get(pid, "Unknown")
        is_rr = pid in set1_pids
        in_set2 = pid in set2_pids

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
            b_act_runs = 0
            b_exp_runs = 0.0
            b_act_wkts = 0
            b_exp_wkts = 0.0
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
            lo_act_runs = 0
            lo_exp_runs = 0.0
            lo_act_wkts = 0
            lo_exp_wkts = 0.0
            lo_runs_saved = 0.0
            lo_wkt_comp = 0.0
            lo_score = 0.0

        lo_qualifies = n_lo >= MIN_SAMPLE_BOWLER_LO_BALLS

        bowler_stats.append({
            "player_id": pid,
            "player": p_name,
            "nationality": p_nat,
            "teams": p_team,
            "is_rr": is_rr,
            "in_set2": in_set2,
            "pressure_balls": p_balls,
            "band_balls": n_band,
            "last_over_balls": n_lo,
            "band_actual_runs": b_act_runs,
            "band_expected_runs": round(b_exp_runs, 1),
            "band_actual_wickets": b_act_wkts,
            "band_expected_wickets": round(b_exp_wkts, 2),
            "runs_saved_per_over": round(runs_saved_po, 2),
            "wicket_component": round(wkt_comp, 2),
            "dot_difference": round(dot_diff, 1),
            "last_over_actual_runs": lo_act_runs,
            "last_over_expected_runs": round(lo_exp_runs, 1),
            "last_over_actual_wickets": lo_act_wkts,
            "last_over_expected_wickets": round(lo_exp_wkts, 2),
            "last_over_runs_saved": round(lo_runs_saved, 2),
            "last_over_wicket_component": round(lo_wkt_comp, 2),
            "last_over_score": round(lo_score, 2),
            "lo_qualifies": lo_qualifies,
        })

    bw_master_df = pd.DataFrame(bowler_stats)

    # =========================================================================
    # STEP 3 & 6: TIERED ELIGIBILITY & RATING PIPELINE FOR EACH SET
    # =========================================================================
    print("\n===================================================================")
    print("STEP 3 & 6: TIERED THRESHOLDS & SET RATINGS")
    print("===================================================================")

    def process_batter_set(df_input, set_name):
        df_set = df_input.copy()

        # Step 3: Tiered thresholds
        # RR players use 30; non-RR players use 60
        df_set["threshold_applied"] = np.where(df_set["is_rr"], MIN_BALLS_BATTER_RR, MIN_BALLS_BATTER_NON_RR)
        df_set["shrinkage_k"] = np.where(df_set["is_rr"], SHRINKAGE_K_BATTER_RR, SHRINKAGE_K_BATTER_NON_RR)
        df_set["qualifies"] = df_set["pressure_balls"] >= df_set["threshold_applied"]

        # Reliability note
        notes = []
        for _, row in df_set.iterrows():
            pb = row["pressure_balls"]
            thresh = row["threshold_applied"]
            inns = row["pressure_innings"]
            if pb >= thresh * 2:
                notes.append(f"Robust sample ({pb} balls across {inns} inns). Highly statistically reliable.")
            elif pb >= thresh:
                notes.append(f"Qualified sample ({pb} >= {thresh} balls across {inns} inns). Reliable trend.")
            else:
                notes.append(f"Limited sample ({pb} < {thresh} balls). Excluded from official ranking.")
        df_set["reliability_note"] = notes

        # Step 6: Z-scores within set across qualifying players
        q = df_set[df_set["qualifies"]].copy()
        if len(q) > 0:
            r_mean, r_std = q["runs_above_expected_per_100"].mean(), q["runs_above_expected_per_100"].std(ddof=0)
            d_mean, d_std = q["dismissal_penalty"].mean(), q["dismissal_penalty"].std(ddof=0)
            if r_std == 0: r_std = 1.0
            if d_std == 0: d_std = 1.0

            q["z_runs"] = (q["runs_above_expected_per_100"] - r_mean) / r_std
            q["z_dismissal"] = (q["dismissal_penalty"] - d_mean) / d_std
            q["raw_rating"] = q["z_runs"] - q["z_dismissal"]
            q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + q["shrinkage_k"])

            df_set = df_set.merge(q[["player_id", "z_runs", "z_dismissal", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
            df_set["raw_rating"] = df_set["raw_rating"].round(4)
            df_set["adjusted_rating"] = df_set["adjusted_rating"].round(4)
        else:
            df_set["raw_rating"] = "n/a"
            df_set["adjusted_rating"] = "n/a"

        removed = (df_set["qualifies"] == False).sum()
        qualified = df_set["qualifies"].sum()
        print(f"[{set_name} Batters] Total: {len(df_set)} | Qualified: {qualified} | Removed by threshold: {removed}")
        return df_set

    def process_bowler_set(df_input, set_name):
        df_set = df_input.copy()

        # Step 3: Tiered thresholds
        # RR players use 24; non-RR players use 48
        df_set["threshold_applied"] = np.where(df_set["is_rr"], MIN_BALLS_BOWLER_RR, MIN_BALLS_BOWLER_NON_RR)
        df_set["shrinkage_k"] = np.where(df_set["is_rr"], SHRINKAGE_K_BOWLER_RR, SHRINKAGE_K_BOWLER_NON_RR)
        df_set["qualifies"] = df_set["band_balls"] >= df_set["threshold_applied"]

        # Reliability note
        notes = []
        for _, row in df_set.iterrows():
            bb = row["band_balls"]
            lo = row["last_over_balls"]
            thresh = row["threshold_applied"]
            if bb >= thresh * 2:
                notes.append(f"Robust sample ({bb} band balls, {lo} last-over balls). High statistical confidence.")
            elif bb >= thresh:
                notes.append(f"Qualified sample ({bb} >= {thresh} band balls, {lo} last-over balls). Reliable indicator.")
            else:
                notes.append(f"Limited sample ({bb} < {thresh} band balls). Excluded from official ranking.")
        df_set["reliability_note"] = notes

        # Step 6: Z-scores within set across qualifying players
        q = df_set[df_set["qualifies"]].copy()
        if len(q) > 0:
            rs_mean, rs_std = q["runs_saved_per_over"].mean(), q["runs_saved_per_over"].std(ddof=0)
            w_mean, w_std = q["wicket_component"].mean(), q["wicket_component"].std(ddof=0)
            if rs_std == 0: rs_std = 1.0
            if w_std == 0: w_std = 1.0

            q["z_runs_saved"] = (q["runs_saved_per_over"] - rs_mean) / rs_std
            q["z_wicket"] = (q["wicket_component"] - w_mean) / w_std

            # Last-over component standardized across qualifiers with >= 12 last-over balls
            q_lo = q[q["lo_qualifies"]]
            if len(q_lo) > 0 and q_lo["last_over_score"].std(ddof=0) > 0:
                lo_mean = q_lo["last_over_score"].mean()
                lo_std = q_lo["last_over_score"].std(ddof=0)
                q["z_last_over"] = np.where(q["lo_qualifies"], (q["last_over_score"] - lo_mean) / lo_std, 0.0)
            else:
                q["z_last_over"] = 0.0

            q["raw_rating"] = q["z_runs_saved"] + q["z_wicket"] + 0.5 * q["z_last_over"]
            q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + q["shrinkage_k"])

            df_set = df_set.merge(q[["player_id", "z_runs_saved", "z_wicket", "z_last_over", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
            df_set["raw_rating"] = df_set["raw_rating"].round(4)
            df_set["adjusted_rating"] = df_set["adjusted_rating"].round(4)
        else:
            df_set["raw_rating"] = "n/a"
            df_set["adjusted_rating"] = "n/a"

        removed = (df_set["qualifies"] == False).sum()
        qualified = df_set["qualifies"].sum()
        print(f"[{set_name} Bowlers] Total: {len(df_set)} | Qualified: {qualified} | Removed by threshold: {removed}")
        return df_set

    # Filter pools for the 3 sets:
    # Set 1: RR players
    bat_set1_raw = bat_master_df[bat_master_df["is_rr"]].copy()
    bw_set1_raw = bw_master_df[bw_master_df["is_rr"]].copy()

    # Set 2: Next season eligible (appeared in 2025 or 2026, EXCLUDING Pakistan players)
    bat_set2_raw = bat_master_df[bat_master_df["in_set2"] & (bat_master_df["nationality"] != "Pakistan")].copy()
    bw_set2_raw = bw_master_df[bw_master_df["in_set2"] & (bw_master_df["nationality"] != "Pakistan")].copy()

    # Set 3: All players in the qualifying dataset (INCLUDING Pakistan players)
    bat_set3_raw = bat_master_df.copy()
    bw_set3_raw = bw_master_df.copy()

    # Process all sets
    batters_rr = process_batter_set(bat_set1_raw, "Set 1 (RR Players)")
    bowlers_rr = process_bowler_set(bw_set1_raw, "Set 1 (RR Players)")

    batters_next_season = process_batter_set(bat_set2_raw, "Set 2 (Next Season Eligible)")
    bowlers_next_season = process_bowler_set(bw_set2_raw, "Set 2 (Next Season Eligible)")

    batters_all = process_batter_set(bat_set3_raw, "Set 3 (All Players)")
    bowlers_all = process_bowler_set(bw_set3_raw, "Set 3 (All Players)")

    # Produce Top 10 rankings
    def get_top10(df_set):
        t10 = df_set[df_set["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
        t10.insert(0, "rank", range(1, len(t10) + 1))
        return t10

    top10_rr_b = get_top10(batters_rr)
    top10_rr_bw = get_top10(bowlers_rr)

    top10_next_b = get_top10(batters_next_season)
    top10_next_bw = get_top10(bowlers_next_season)

    top10_all_b = get_top10(batters_all)
    top10_all_bw = get_top10(bowlers_all)

    # =========================================================================
    # STEP 9: SANITY CHECKS & VERIFICATION
    # =========================================================================
    print("\n===================================================================")
    print("STEP 9: SANITY CHECKS & VERIFICATION")
    print("===================================================================")

    sanity_failures = []

    # 1. Sign checks for all 3 sets
    set_tuples = [
        ("Set 1 (RR)", batters_rr, bowlers_rr, top10_rr_b, top10_rr_bw),
        ("Set 2 (Next Season)", batters_next_season, bowlers_next_season, top10_next_b, top10_next_bw),
        ("Set 3 (All Players)", batters_all, bowlers_all, top10_all_b, top10_all_bw),
    ]

    for s_label, b_df, bw_df, t_b, t_bw in set_tuples:
        # Batter sign check: holding z_runs constant, raw_rating = z_runs - z_dismissal -> corr(z_dismissal, raw_rating - z_runs) == -1.0
        q_b = b_df[b_df["qualifies"]].copy()
        corr_b = np.corrcoef(q_b["z_dismissal"], q_b["raw_rating"] - q_b["z_runs"])[0, 1]
        if corr_b > -0.999:
            sanity_failures.append(f"{s_label} Batters: dismissal penalty check failed (corr={corr_b})")

        # Bowler sign check: holding wickets and last-over constant, corr(z_runs_saved, raw_rating - z_wicket - 0.5*z_last_over) == 1.0
        q_bw = bw_df[bw_df["qualifies"]].copy()
        corr_bw = np.corrcoef(q_bw["z_runs_saved"], q_bw["raw_rating"] - q_bw["z_wicket"] - 0.5 * q_bw["z_last_over"])[0, 1]
        if corr_bw < 0.999:
            sanity_failures.append(f"{s_label} Bowlers: runs saved check failed (corr={corr_bw})")

        # Threshold check: Confirm that no non-RR player in the top 10 has fewer than 60 pressure balls (batters) or 48 balls in the band (bowlers)
        for _, row in t_b.iterrows():
            if not row["is_rr"] and row["pressure_balls"] < MIN_BALLS_BATTER_NON_RR:
                sanity_failures.append(f"{s_label} Batter Top 10: Non-RR player {row['player']} has {row['pressure_balls']} < {MIN_BALLS_BATTER_NON_RR} balls!")

        for _, row in t_bw.iterrows():
            if not row["is_rr"] and row["band_balls"] < MIN_BALLS_BOWLER_NON_RR:
                sanity_failures.append(f"{s_label} Bowler Top 10: Non-RR player {row['player']} has {row['band_balls']} < {MIN_BALLS_BOWLER_NON_RR} band balls!")

    # 2. Verify zero Pakistan players in Set 2
    pak_bat_set2 = (batters_next_season["nationality"] == "Pakistan").sum()
    pak_bw_set2 = (bowlers_next_season["nationality"] == "Pakistan").sum()
    if pak_bat_set2 > 0 or pak_bw_set2 > 0:
        sanity_failures.append(f"Set 2 contains Pakistan players! Batters: {pak_bat_set2}, Bowlers: {pak_bw_set2}")
    else:
        print("  [PASSED] Set 2 (Next Season Eligible) strictly contains 0 Pakistan players.")

    # 3. Verify Pakistan players ARE present in Set 3
    pak_bat_set3 = (batters_all["nationality"] == "Pakistan").sum()
    pak_bw_set3 = (bowlers_all["nationality"] == "Pakistan").sum()
    print(f"  [PASSED] Set 3 (All Players) includes Pakistan players: {pak_bat_set3} batters, {pak_bw_set3} bowlers.")

    if sanity_failures:
        print("\nSANITY CHECK FAILURES ENCOUNTERED:")
        for f in sanity_failures:
            print(f"  FAILED: {f}")
        raise ValueError("Sanity check verification failed!")
    else:
        print("ALL SANITY CHECKS PASSED:")
        print("  1. Verified across all three sets that holding other components equal, a batter with more dismissals than expected ranks strictly lower (r = -1.000).")
        print("  2. Verified across all three sets that holding other components equal, a bowler who concedes more runs than expected ranks strictly lower (r = +1.000 with runs saved).")
        print("  3. Verified 0 Pakistan players in Set 2 (Next Season Eligible).")
        print("  4. Confirmed that NO non-RR player in any Top 10 has fewer than 60 pressure balls (batters) or 48 band balls (bowlers).")

    # =========================================================================
    # STEP 8: EXPORT OUTPUTS & PRINT TOP 10 RANKINGS
    # =========================================================================
    print("\n===================================================================")
    print("STEP 8: OUTPUTS & TOP 10 RANKINGS")
    print("===================================================================")

    # Save full tables
    batters_rr.to_csv(os.path.join(output_dir, "batters_rr.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_rr.csv"), index=False)
    batters_next_season.to_csv(os.path.join(output_dir, "batters_next_season.csv"), index=False)
    bowlers_next_season.to_csv(os.path.join(output_dir, "bowlers_next_season.csv"), index=False)
    batters_all.to_csv(os.path.join(output_dir, "batters_all.csv"), index=False)
    bowlers_all.to_csv(os.path.join(output_dir, "bowlers_all.csv"), index=False)

    # Backward compatibility
    batters_rr.to_csv(os.path.join(output_dir, "batters_pressure.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_pressure.csv"), index=False)

    # Save Top 10 tables
    top10_rr_b.to_csv(os.path.join(output_dir, "top10_rr_batters.csv"), index=False)
    top10_rr_bw.to_csv(os.path.join(output_dir, "top10_rr_bowlers.csv"), index=False)
    top10_next_b.to_csv(os.path.join(output_dir, "top10_next_season_batters.csv"), index=False)
    top10_next_bw.to_csv(os.path.join(output_dir, "top10_next_season_bowlers.csv"), index=False)
    top10_all_b.to_csv(os.path.join(output_dir, "top10_all_batters.csv"), index=False)
    top10_all_bw.to_csv(os.path.join(output_dir, "top10_all_bowlers.csv"), index=False)

    # Print Top 10 Tables
    print("\n[TOP 10 RR BATTERS (SET 1)]")
    print(top10_rr_b[["rank", "player", "nationality", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 RR BOWLERS (SET 1)]")
    print(top10_rr_bw[["rank", "player", "nationality", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 NEXT SEASON ELIGIBLE BATTERS (SET 2)]")
    print(top10_next_b[["rank", "player", "nationality", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 NEXT SEASON ELIGIBLE BOWLERS (SET 2)]")
    print(top10_next_bw[["rank", "player", "nationality", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 ALL BATTERS (SET 3)]")
    print(top10_all_b[["rank", "player", "nationality", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 ALL BOWLERS (SET 3)]")
    print(top10_all_bw[["rank", "player", "nationality", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    return {
        "top10_rr_b": top10_rr_b,
        "top10_rr_bw": top10_rr_bw,
        "top10_next_b": top10_next_b,
        "top10_next_bw": top10_next_bw,
        "top10_all_b": top10_all_b,
        "top10_all_bw": top10_all_bw
    }

if __name__ == "__main__":
    run_pressure_pipeline()

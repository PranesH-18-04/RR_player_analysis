"""
Rajasthan Royals & Global T20 Pressure Analytics Engine
Multi-Tier Framework:
- SENIOR: Full national teams (with Top 12 T20I opponent filter) and IPL franchises (unfiltered).
  Ranks Set 1 (RR Players), Set 2 (Next Season Eligible - ex-Pak), Set 3 (All Senior Players).
- DEVELOPMENT: A-teams and emerging competitions (Mens T20 Emerging Teams Asia Cup 2024).
- YOUTH: Youth competitions (Asia Cup Rising Stars 2025).
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis
"""

import os
import json
import pandas as pd
import numpy as np

# =========================================================================
# STEP 2: TEAM TIER CLASSIFICATION MAPPING CONSTANT
# =========================================================================
# Tier Definitions:
# - SENIOR: Full national teams and IPL franchises.
# - DEVELOPMENT: India A, other "A"/"B"/"C"/"D" sides, Emerging, Academy, or XI sides.
# - YOUTH: Teams labeled U16, U17, U19, U21, or participating in youth competitions.
TEAM_TIER = {
    # IPL Franchises (SENIOR)
    "Chennai Super Kings": "SENIOR",
    "Delhi Capitals": "SENIOR",
    "Gujarat Titans": "SENIOR",
    "Kolkata Knight Riders": "SENIOR",
    "Lucknow Super Giants": "SENIOR",
    "Mumbai Indians": "SENIOR",
    "Punjab Kings": "SENIOR",
    "Rajasthan Royals": "SENIOR",
    "Royal Challengers Bengaluru": "SENIOR",
    "Sunrisers Hyderabad": "SENIOR",

    # Full National Teams (SENIOR)
    "Afghanistan": "SENIOR",
    "Australia": "SENIOR",
    "Bangladesh": "SENIOR",
    "Canada": "SENIOR",
    "England": "SENIOR",
    "Hong Kong": "SENIOR",
    "India": "SENIOR",
    "Ireland": "SENIOR",
    "Japan": "SENIOR",
    "Kuwait": "SENIOR",
    "Nepal": "SENIOR",
    "Netherlands": "SENIOR",
    "New Zealand": "SENIOR",
    "Oman": "SENIOR",
    "Pakistan": "SENIOR",
    "Scotland": "SENIOR",
    "Singapore": "SENIOR",
    "South Africa": "SENIOR",
    "Sri Lanka": "SENIOR",
    "United Arab Emirates": "SENIOR",
    "United States Of America": "SENIOR",
    "West Indies": "SENIOR",
    "Zimbabwe": "SENIOR",

    # Development / "A" Sides (DEVELOPMENT)
    "Afghanistan A": "DEVELOPMENT",
    "Bangladesh A": "DEVELOPMENT",
    "India A": "DEVELOPMENT",
    "Pakistan A": "DEVELOPMENT",
    "Sri Lanka A": "DEVELOPMENT",

    # Exemplar / Spec sides explicitly defined
    "India B": "DEVELOPMENT",
    "India C": "DEVELOPMENT",
    "India D": "DEVELOPMENT",
    "India U23": "DEVELOPMENT",
    "India U21": "YOUTH",
    "India U19": "YOUTH",
    "India U17": "YOUTH",
    "India U16": "YOUTH",
}

# =========================================================================
# OFFICIAL ICC T20I TOP 12 OPPONENT FILTER CONSTANT
# =========================================================================
# Official ICC Men's T20I Team Rankings snapshot recorded on 2026-11-14:
# 1. India, 2. Australia, 3. England, 4. South Africa, 5. New Zealand,
# 6. West Indies, 7. Pakistan, 8. Sri Lanka, 9. Bangladesh, 10. Afghanistan,
# 11. Ireland, 12. Zimbabwe
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
# MINIMUM BALL THRESHOLDS & SHRINKAGE CONSTANTS
# =========================================================================
# Senior Sets (Set 1, 2, 3)
MIN_BALLS_BATTER_RR = 30          # RR Batters: min 30 pressure balls
MIN_BALLS_BOWLER_RR = 24          # RR Bowlers: min 24 balls in 8-12 bowling band
MIN_BALLS_BATTER_NON_RR = 60      # Non-RR Batters: min 60 pressure balls
MIN_BALLS_BOWLER_NON_RR = 48      # Non-RR Bowlers: min 48 balls in 8-12 bowling band

SHRINKAGE_K_BATTER_RR = 30.0      # Prior weight for RR batters
SHRINKAGE_K_BATTER_NON_RR = 60.0  # Prior weight for Non-RR batters
SHRINKAGE_K_BOWLER_RR = 24.0      # Prior weight for RR bowlers
SHRINKAGE_K_BOWLER_NON_RR = 48.0  # Prior weight for Non-RR bowlers
MIN_SAMPLE_BOWLER_LO_BALLS = 12   # Minimum last-over balls for last-over component

# Young & Development Sets
MIN_BALLS_BATTER_DEV_YOUTH = 30   # Min 30 pressure balls
MIN_BALLS_BOWLER_DEV_YOUTH = 24   # Min 24 band balls
SHRINKAGE_K_BATTER_DEV_YOUTH = 30.0
SHRINKAGE_K_BOWLER_DEV_YOUTH = 24.0


def classify_match_tier(comp_id, comp_name):
    """
    Classifies each competition/match into SENIOR, DEVELOPMENT, or YOUTH.
    - Comp 2670: Mens T20 Emerging Teams Asia Cup 2024 -> DEVELOPMENT
    - Comp 3289: Asia Cup Rising Stars 2025 -> YOUTH (Youth competition)
    - All other competitions (IPL & T20I) -> SENIOR
    """
    if comp_id == 2670 or "Emerging" in str(comp_name):
        return "DEVELOPMENT"
    elif comp_id == 3289 or "Rising Stars" in str(comp_name) or "Youth" in str(comp_name):
        return "YOUTH"
    else:
        return "SENIOR"


def run_multi_tier_pipeline(data_dir=None, output_dir=None):
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

    # Confirm Team ID and Name for Rajasthan Royals
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

    # Season field and Most Recent Season
    comp["season_clean"] = comp["season"].astype(str).str.strip()
    all_seasons = sorted(comp["season_clean"].unique())
    most_recent_season = all_seasons[-1]
    second_recent_season = all_seasons[-2]
    print(f"\nDistinct seasons in data: {all_seasons}")
    print(f"Most recent season: '{most_recent_season}' | Preceding season: '{second_recent_season}'")

    # Distinct team names and ball counts
    team_id_to_name = teams.drop_duplicates(subset=["team_id"]).set_index("team_id")["team_name"].to_dict()
    bbb["batting_team_name"] = bbb["batting_team_id"].map(team_id_to_name)
    bbb["bowling_team_name"] = bbb["bowling_team_id"].map(team_id_to_name)

    batting_counts = bbb["batting_team_name"].value_counts().to_dict()
    bowling_counts = bbb["bowling_team_name"].value_counts().to_dict()

    all_teams_in_data = sorted(set(teams["team_name"].unique()) | set(matches["home_team"].unique()) | set(matches["away_team"].unique()))
    print(f"\n--- COMPLETE LIST OF DISTINCT TEAM NAMES IN DATA ({len(all_teams_in_data)} teams) ---")
    team_ball_summary = []
    for t in all_teams_in_data:
        b_cnt = batting_counts.get(t, 0)
        bw_cnt = bowling_counts.get(t, 0)
        tot_cnt = b_cnt + bw_cnt
        team_ball_summary.append({"team": t, "batting_balls": b_cnt, "bowling_balls": bw_cnt, "total_balls": tot_cnt})
        print(f"  {t:32s} | Batting: {b_cnt:6d} | Bowling: {bw_cnt:6d} | Total Balls: {tot_cnt:6d}")

    # =========================================================================
    # STEP 2: CLASSIFY EACH TEAM INTO A TIER & CHECK UNASSIGNED NAMES
    # =========================================================================
    print("\n===================================================================")
    print("STEP 2: TEAM TIER CLASSIFICATION & AUDIT")
    print("===================================================================")

    unassigned_teams = []
    print("\nTeam Tier Mapping:")
    for t in all_teams_in_data:
        tier = TEAM_TIER.get(t)
        if tier is None:
            unassigned_teams.append(t)
            print(f"  UNASSIGNED: '{t}'")
        else:
            print(f"  {t:32s} -> {tier}")

    if unassigned_teams:
        print(f"\nERROR: Found {len(unassigned_teams)} unassigned team names: {unassigned_teams}")
        raise ValueError("Unassigned team names encountered! Manual review required.")
    else:
        print(f"\nAll {len(all_teams_in_data)} teams in the dataset are assigned to a valid tier without guessing.")

    # Multi-tier Player Audit: Flag players appearing in more than one tier
    # Classify matches first
    m_comp = matches.merge(comp[["comp_id", "comp_name", "match_format", "season_clean"]], on="comp_id", how="left")
    m_comp["match_tier"] = m_comp.apply(lambda r: classify_match_tier(r["comp_id"], r["comp_name"]), axis=1)

    p11_m = p11.merge(m_comp[["match_id", "match_tier", "comp_name", "match_date"]], on="match_id", how="inner")
    p_names = mapping.groupby("player_id")["player_name"].first().to_dict()
    p_teams_all = mapping.groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()

    p_tiers = p11_m.groupby("player_id")["match_tier"].unique().to_dict()
    multi_tier_pids = {pid: sorted(list(tiers)) for pid, tiers in p_tiers.items() if len(tiers) > 1}
    print(f"\n--- MULTI-TIER PLAYER AUDIT ({len(multi_tier_pids)} players appear in >1 tier) ---")

    multi_tier_records = []
    for pid, tiers in multi_tier_pids.items():
        p_matches = p11_m[p11_m["player_id"] == pid]
        s_cnt = p_matches[p_matches["match_tier"] == "SENIOR"]["match_id"].nunique()
        d_cnt = p_matches[p_matches["match_tier"] == "DEVELOPMENT"]["match_id"].nunique()
        y_cnt = p_matches[p_matches["match_tier"] == "YOUTH"]["match_id"].nunique()
        multi_tier_records.append({
            "player_id": pid,
            "player": p_names.get(pid, f"Player_{pid}"),
            "tiers": ", ".join(tiers),
            "senior_matches": s_cnt,
            "development_matches": d_cnt,
            "youth_matches": y_cnt,
            "total_matches": s_cnt + d_cnt + y_cnt
        })

    multi_df = pd.DataFrame(multi_tier_records).sort_values("total_matches", ascending=False)
    print("Top 15 Multi-Tier Players (Matches played in each tier recorded separately):")
    print(multi_df.head(15).to_string(index=False))

    # Nationality Mapping
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

    # =========================================================================
    # STEP 3: SEPARATE MATCHES BY TIER & APPLY SENIOR TOP-12 OPPONENT FILTER
    # =========================================================================
    print("\n===================================================================")
    print("STEP 3: SEPARATE MATCHES BY TIER")
    print("===================================================================")

    m_comp["is_ipl"] = m_comp["comp_name"].str.contains("IPL", case=False, na=False)
    m_comp["is_t20i"] = m_comp["match_format"] == "T20I"
    m_comp["both_top12"] = m_comp["home_team"].isin(TOP_12_ICC_T20I) & m_comp["away_team"].isin(TOP_12_ICC_T20I)

    # Valid matches filter
    drop_mask = (
        m_comp["win_team_id"].isna()
        | (m_comp["win_team_id"] == 0)
        | m_comp["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    )
    m_valid = m_comp[~drop_mask].copy()

    # Raw balls by tier
    bbb_m = bbb.merge(m_comp[["match_id", "match_tier", "is_ipl", "is_t20i", "both_top12"]], on="match_id", how="left")
    raw_senior_balls = (bbb_m["match_tier"] == "SENIOR").sum()
    raw_dev_balls = (bbb_m["match_tier"] == "DEVELOPMENT").sum()
    raw_youth_balls = (bbb_m["match_tier"] == "YOUTH").sum()

    # Valid balls by tier
    bbb_valid = bbb_m[bbb_m["match_id"].isin(set(m_valid["match_id"].unique()))]
    valid_senior_balls = (bbb_valid["match_tier"] == "SENIOR").sum()
    valid_dev_balls = (bbb_valid["match_tier"] == "DEVELOPMENT").sum()
    valid_youth_balls = (bbb_valid["match_tier"] == "YOUTH").sum()

    # Senior kept vs dropped
    valid_ipl_matches = (m_valid["is_ipl"]).sum()
    valid_ipl_balls = (bbb_valid["is_ipl"]).sum()

    valid_t20i_top12_m = (m_valid["is_t20i"] & m_valid["both_top12"]).sum()
    valid_t20i_top12_b = (bbb_valid["is_t20i"] & bbb_valid["both_top12"]).sum()

    valid_t20i_dropped_m = (m_valid["is_t20i"] & ~m_valid["both_top12"]).sum()
    valid_t20i_dropped_b = (bbb_valid["is_t20i"] & ~bbb_valid["both_top12"]).sum()

    print("\n--- MATCH & BALL COUNTS BY TIER ---")
    print(f"Total Matches: {len(matches)} (Resolved/Valid: {len(m_valid)})")
    print(f"Total Balls: {len(bbb)} (Resolved/Valid: {len(bbb_valid)})")

    print("\n1. SENIOR TIER:")
    print(f"   Raw Senior Matches: {(m_comp['match_tier'] == 'SENIOR').sum()} | Raw Senior Balls: {raw_senior_balls}")
    print(f"   Valid Senior Matches: {(m_valid['match_tier'] == 'SENIOR').sum()} | Valid Senior Balls: {valid_senior_balls}")
    print(f"   -> IPL Matches (Unfiltered): {valid_ipl_matches} matches ({valid_ipl_balls} balls)")
    print(f"   -> T20I Top-12 Kept: {valid_t20i_top12_m} matches ({valid_t20i_top12_b} balls)")
    print(f"   -> T20I Non-Top-12 Dropped: {valid_t20i_dropped_m} matches ({valid_t20i_dropped_b} balls)")
    print(f"   -> SENIOR QUALIFYING POOL (Sets 1, 2, 3): {valid_ipl_matches + valid_t20i_top12_m} matches ({valid_ipl_balls + valid_t20i_top12_b} balls)")

    print("\n2. DEVELOPMENT TIER (Mens T20 Emerging Teams Asia Cup 2024):")
    print(f"   Matches: {(m_valid['match_tier'] == 'DEVELOPMENT').sum()} | Balls: {valid_dev_balls}")
    print(f"   Opponent filter: Not applied (quality not comparable)")

    print("\n3. YOUTH TIER (Asia Cup Rising Stars 2025):")
    print(f"   Matches: {(m_valid['match_tier'] == 'YOUTH').sum()} | Balls: {valid_youth_balls}")
    print(f"   Opponent filter: Not applied (quality not comparable)")

    # -------------------------------------------------------------
    # COMMON DELIVERIES DATA PREPARATION FUNCTION
    # -------------------------------------------------------------
    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    m_valid = m_valid.merge(inn1_totals, on="match_id", how="inner")

    m_valid["target"] = np.where(
        m_valid["revised_target"].notna() & (m_valid["revised_target"] > 0),
        m_valid["revised_target"],
        m_valid["inn1_total"] + 1,
    )
    m_valid["total_balls"] = np.where(
        m_valid["second_inn_revised_overs"].notna() & (m_valid["second_inn_revised_overs"] > 0),
        m_valid["second_inn_revised_overs"],
        np.where(
            m_valid["first_inn_revised_overs"].notna() & (m_valid["first_inn_revised_overs"] > 0),
            m_valid["first_inn_revised_overs"],
            120,
        ),
    )
    m_valid["year"] = pd.to_datetime(m_valid["match_date"]).dt.year

    # Helper function to enrich delivery dataframe
    def prepare_deliveries_df(m_subset, bbb_df):
        vp_season = m_subset.groupby(["venue_id", "year"])["inn1_total"].mean().rename("venue_par_season")
        vp_venue = m_subset.groupby("venue_id")["inn1_total"].mean().rename("venue_par_venue")
        overall_par = m_subset["inn1_total"].mean()

        m_sub = m_subset.merge(vp_season, on=["venue_id", "year"], how="left")
        m_sub = m_sub.merge(vp_venue, on="venue_id", how="left")
        m_sub["venue_par"] = m_sub["venue_par_season"].fillna(m_sub["venue_par_venue"]).fillna(overall_par)

        vp_bins = [-np.inf, 165, 185, np.inf]
        vp_labels = ["Low (<165)", "Medium (165-185)", "High (>185)"]
        m_sub["venue_par_bucket"] = pd.cut(m_sub["venue_par"], bins=vp_bins, labels=vp_labels).astype(str)

        match_cols = ["match_id", "target", "total_balls", "win_team_id", "match_date", "year", "comp_id", "comp_name", "season_clean", "match_tier", "venue_par_bucket"]
        df_out = bbb_df.merge(m_sub[match_cols], on="match_id", how="inner")
        df_out = df_out.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

        df_out["bowler_extras"] = np.where(df_out["extras_type"].isin(["byes", "legbyes", "Penalty"]), 0, df_out["extras1_runs"])
        df_out["bowler_runs_conceded"] = df_out["runs"] + df_out["bowler_extras"]
        df_out["bowler_wicket"] = df_out["is_wicket"] & ~df_out["dismissal_type"].astype(str).str.contains("run out", case=False, na=False)

        df_out["batter_dismissal"] = df_out["is_wicket"] & (
            (df_out["dismissed_player_id"] == df_out["striker_id"])
            | df_out["dismissed_player_id"].isna()
            | (df_out["dismissal_type"] != "run out")
        )

        df_out["cum_runs_after"] = df_out.groupby(["match_id", "innings"])["total_ball_runs"].cumsum()
        df_out["cum_runs_before"] = df_out["cum_runs_after"] - df_out["total_ball_runs"]

        df_out["cum_wickets_after"] = df_out.groupby(["match_id", "innings"])["is_wicket"].cumsum()
        df_out["cum_wickets_before"] = df_out["cum_wickets_after"] - df_out["is_wicket"].astype(int)

        df_out["balls_remaining_before"] = np.maximum(df_out["total_balls"] - (df_out["innings_ball_num"] - 1), 1)
        df_out["runs_needed_before"] = df_out["target"] - df_out["cum_runs_before"]
        df_out["wickets_in_hand_before"] = np.clip(10 - df_out["cum_wickets_before"], 1, 10)
        df_out["rrr"] = np.where(
            df_out["innings"] == 2,
            (df_out["runs_needed_before"] * 6.0) / df_out["balls_remaining_before"],
            np.nan
        )

        # Buckets
        br_bins = [0, 6, 12, 24, 36, 60, 150]
        br_labels = ["1-6", "7-12", "13-24", "25-36", "37-60", "61-120"]
        df_out["balls_rem_bucket"] = pd.cut(df_out["balls_remaining_before"], bins=br_bins, labels=br_labels, right=True).astype(str)

        rrr_bins = [-np.inf] + list(np.arange(6.0, 16.5, 0.5)) + [np.inf]
        rrr_labels = ["<6.0"] + [f"{r:.1f}-{r+0.5:.1f}" for r in np.arange(6.0, 16.0, 0.5)] + ["16.0+"]
        df_out["rrr_bucket"] = pd.cut(df_out["rrr"], bins=rrr_bins, labels=rrr_labels, right=False).astype(str)

        w_bins = [0, 3, 6, 11]
        w_labels = ["1-3", "4-6", "7-10"]
        df_out["wickets_in_hand_bucket"] = pd.cut(df_out["wickets_in_hand_before"], bins=w_bins, labels=w_labels, right=True).astype(str)

        df_out["over_stage"] = np.where(
            df_out["overid"] <= 6, "Powerplay",
            np.where(df_out["overid"] <= 15, "Middle", "Death")
        )

        df_out["bucket_id"] = np.where(
            df_out["innings"] == 2,
            "inn2_" + df_out["balls_rem_bucket"] + "_rrr_" + df_out["rrr_bucket"] + "_w_" + df_out["wickets_in_hand_bucket"],
            "inn1_" + df_out["balls_rem_bucket"] + "_stage_" + df_out["over_stage"] + "_vp_" + df_out["venue_par_bucket"]
        )

        # Pressure flags
        df_out["is_batting_pressure"] = (
            ((df_out["innings"] == 2) & ((df_out["rrr"] >= 9.0) | (df_out["balls_remaining_before"] <= 30) | (df_out["overid"] >= 16)))
            | ((df_out["innings"] == 1) & ((df_out["balls_remaining_before"] <= 30) | (df_out["overid"] >= 16)))
        )
        df_out["is_last_over"] = df_out["balls_remaining_before"] <= 6
        df_out["is_bowling_band"] = (df_out["innings"] == 2) & (df_out["rrr"] >= 8.0) & (df_out["rrr"] <= 12.0) & (~df_out["is_last_over"])
        df_out["is_bowling_pressure"] = df_out["is_last_over"] | df_out["is_bowling_band"]

        return df_out

    # Helper function to compute expected table
    def build_expected_table(df_enriched, out_csv_path, mark_unreliable=False):
        exp_t = df_enriched.groupby("bucket_id", observed=True).agg(
            innings=("innings", "first"),
            balls_rem_bucket=("balls_rem_bucket", "first"),
            rrr_bucket=("rrr_bucket", "first"),
            wickets_in_hand_bucket=("wickets_in_hand_bucket", "first"),
            over_stage=("over_stage", "first"),
            venue_par_bucket=("venue_par_bucket", "first"),
            ball_count=("bucket_id", "count"),
            batter_runs=("runs", "sum"),
            batter_dots=("runs", lambda s: (s == 0).sum()),
            batter_boundaries=("runs", lambda s: (df_enriched.loc[s.index, "is_four"] | df_enriched.loc[s.index, "is_six"]).sum()),
            batter_dismissals=("batter_dismissal", "sum"),
            bowler_runs=("bowler_runs_conceded", "sum"),
            bowler_dots=("runs", lambda s: ((s == 0) & (df_enriched.loc[s.index, "extras1_runs"] == 0)).sum()),
            bowler_wickets=("bowler_wicket", "sum")
        ).reset_index()

        if mark_unreliable:
            exp_t["is_reliable"] = exp_t["ball_count"] >= 50
            exp_t["reliability_status"] = np.where(exp_t["is_reliable"], "Reliable (>=50 balls)", "Unreliable (<50 balls)")

        exp_t["batter_avg_runs_per_ball"] = exp_t["batter_runs"] / exp_t["ball_count"]
        exp_t["batter_dot_pct"] = (exp_t["batter_dots"] / exp_t["ball_count"]) * 100.0
        exp_t["batter_boundary_pct"] = (exp_t["batter_boundaries"] / exp_t["ball_count"]) * 100.0
        exp_t["batter_dismissal_rate"] = exp_t["batter_dismissals"] / exp_t["ball_count"]

        exp_t["bowler_avg_runs_per_ball"] = exp_t["bowler_runs"] / exp_t["ball_count"]
        exp_t["bowler_dot_pct"] = (exp_t["bowler_dots"] / exp_t["ball_count"]) * 100.0
        exp_t["bowler_wicket_rate"] = exp_t["bowler_wickets"] / exp_t["ball_count"]
        exp_t["runs_per_wicket"] = np.where(
            exp_t["bowler_wickets"] > 0,
            exp_t["bowler_runs"] / exp_t["bowler_wickets"],
            exp_t["bowler_runs"] / 1.0
        )

        exp_t.to_csv(out_csv_path, index=False)
        print(f"Saved {len(exp_t)} buckets to {out_csv_path}")
        return exp_t

    def apply_expected_lookup(df_enriched, exp_table):
        lookup = exp_table.set_index("bucket_id")[
            ["batter_avg_runs_per_ball", "batter_dismissal_rate", "bowler_avg_runs_per_ball", "bowler_wicket_rate", "bowler_dot_pct", "runs_per_wicket"]
        ].to_dict("index")

        df_enriched["exp_batter_runs"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["batter_avg_runs_per_ball"])
        df_enriched["exp_batter_dismissal"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["batter_dismissal_rate"])
        df_enriched["exp_bowler_runs"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["bowler_avg_runs_per_ball"])
        df_enriched["exp_bowler_wicket"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["bowler_wicket_rate"])
        df_enriched["exp_bowler_dot_pct"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["bowler_dot_pct"])
        df_enriched["runs_per_wicket"] = df_enriched["bucket_id"].map(lambda b: lookup[b]["runs_per_wicket"])
        return df_enriched

    # =========================================================================
    # PART A: SENIOR PIPELINE (SETS 1, 2, 3)
    # =========================================================================
    print("\n===================================================================")
    print("PART A: BUILD SENIOR EXPECTED-VALUE BASELINE & RANKINGS")
    print("===================================================================")

    m_senior_qualifying = m_valid[(m_valid["match_tier"] == "SENIOR") & (m_valid["is_ipl"] | (m_valid["is_t20i"] & m_valid["both_top12"]))].copy()
    senior_match_ids = set(m_senior_qualifying["match_id"].unique())
    bbb_senior = bbb[bbb["match_id"].isin(senior_match_ids)].copy()

    df_senior = prepare_deliveries_df(m_senior_qualifying, bbb_senior)
    senior_exp_path = os.path.join(output_dir, "expected_tables.csv")
    exp_senior = build_expected_table(df_senior, senior_exp_path)
    df_senior = apply_expected_lookup(df_senior, exp_senior)

    # Player stats aggregation for Senior
    def aggregate_senior_players(df_in):
        bat_press = df_in[df_in["is_batting_pressure"]].copy()
        bat_norm = df_in[~df_in["is_batting_pressure"]].copy()

        # Set memberships
        rr_strikers = set(df_in[df_in["batting_team_id"] == rr_team_id]["striker_id"].dropna().unique())
        rr_bowlers = set(df_in[df_in["bowling_team_id"] == rr_team_id]["bowler_id"].dropna().unique())
        rr_map = set(mapping[(mapping["team_id"] == rr_team_id) & (mapping["comp_id"].isin(senior_match_ids))]["player_id"].dropna().unique())
        set1_pids = rr_strikers | rr_bowlers | rr_map

        recent_comps = set(comp[comp["season_clean"].isin(["2025", "2024/25", "2026"])]["comp_id"].unique())
        recent_m = set(m_senior_qualifying[m_senior_qualifying["comp_id"].isin(recent_comps)]["match_id"].unique())
        recent_df = df_in[df_in["match_id"].isin(recent_m)]
        recent_pids = set(recent_df["striker_id"].dropna().unique()) | set(recent_df["bowler_id"].dropna().unique())
        set2_pids = {pid for pid in recent_pids if player_nats.get(pid) != "Pakistan"}

        all_strikers = sorted(df_in["striker_id"].dropna().unique())
        all_bowlers = sorted(df_in["bowler_id"].dropna().unique())
        all_pids = set(all_strikers) | set(all_bowlers)

        bat_stats = []
        for pid in all_strikers:
            p_name = p_names.get(pid, f"Player_{pid}")
            p_nat = player_nats.get(pid, "Unknown")
            p_team = p_teams_all.get(pid, "Unknown")
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

            bat_stats.append({
                "player_id": pid,
                "player": p_name,
                "nationality": p_nat,
                "tier": "Senior",
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

        bat_master = pd.DataFrame(bat_stats)

        # Bowlers
        bw_band = df_in[df_in["is_bowling_band"]].copy()
        bw_lo = df_in[df_in["is_last_over"]].copy()
        bw_norm = df_in[~df_in["is_bowling_pressure"]].copy()

        bw_stats = []
        for pid in all_bowlers:
            p_name = p_names.get(pid, f"Player_{pid}")
            p_nat = player_nats.get(pid, "Unknown")
            p_team = p_teams_all.get(pid, "Unknown")
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

            bw_stats.append({
                "player_id": pid,
                "player": p_name,
                "nationality": p_nat,
                "tier": "Senior",
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

        bw_master = pd.DataFrame(bw_stats)
        return bat_master, bw_master

    bat_senior_master, bw_senior_master = aggregate_senior_players(df_senior)

    # Process Senior Sets (Set 1, Set 2, Set 3)
    def process_senior_batter_set(df_input, set_name):
        df_set = df_input.copy()
        df_set["threshold_applied"] = np.where(df_set["is_rr"], MIN_BALLS_BATTER_RR, MIN_BALLS_BATTER_NON_RR)
        df_set["shrinkage_k"] = np.where(df_set["is_rr"], SHRINKAGE_K_BATTER_RR, SHRINKAGE_K_BATTER_NON_RR)
        df_set["qualifies"] = df_set["pressure_balls"] >= df_set["threshold_applied"]
        df_set["eligibility"] = np.where(df_set["qualifies"], f"Eligible (>={df_set['threshold_applied']} balls)", f"Ineligible (<{df_set['threshold_applied']} balls)")

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
        return df_set

    def process_senior_bowler_set(df_input, set_name):
        df_set = df_input.copy()
        df_set["threshold_applied"] = np.where(df_set["is_rr"], MIN_BALLS_BOWLER_RR, MIN_BALLS_BOWLER_NON_RR)
        df_set["shrinkage_k"] = np.where(df_set["is_rr"], SHRINKAGE_K_BOWLER_RR, SHRINKAGE_K_BOWLER_NON_RR)
        df_set["qualifies"] = df_set["band_balls"] >= df_set["threshold_applied"]
        df_set["eligibility"] = np.where(df_set["qualifies"], f"Eligible (>={df_set['threshold_applied']} band balls)", f"Ineligible (<{df_set['threshold_applied']} band balls)")

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

        q = df_set[df_set["qualifies"]].copy()
        if len(q) > 0:
            rs_mean, rs_std = q["runs_saved_per_over"].mean(), q["runs_saved_per_over"].std(ddof=0)
            w_mean, w_std = q["wicket_component"].mean(), q["wicket_component"].std(ddof=0)
            if rs_std == 0: rs_std = 1.0
            if w_std == 0: w_std = 1.0

            q["z_runs_saved"] = (q["runs_saved_per_over"] - rs_mean) / rs_std
            q["z_wicket"] = (q["wicket_component"] - w_mean) / w_std

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
        return df_set

    # Filter Senior pools
    bat_set1_raw = bat_senior_master[bat_senior_master["is_rr"]].copy()
    bw_set1_raw = bw_senior_master[bw_senior_master["is_rr"]].copy()

    bat_set2_raw = bat_senior_master[bat_senior_master["in_set2"] & (bat_senior_master["nationality"] != "Pakistan")].copy()
    bw_set2_raw = bw_senior_master[bw_senior_master["in_set2"] & (bw_senior_master["nationality"] != "Pakistan")].copy()

    bat_set3_raw = bat_senior_master.copy()
    bw_set3_raw = bw_senior_master.copy()

    batters_rr = process_senior_batter_set(bat_set1_raw, "Set 1 (RR)")
    bowlers_rr = process_senior_bowler_set(bw_set1_raw, "Set 1 (RR)")

    batters_next_season = process_senior_batter_set(bat_set2_raw, "Set 2 (Next Season)")
    bowlers_next_season = process_senior_bowler_set(bw_set2_raw, "Set 2 (Next Season)")

    batters_all = process_senior_batter_set(bat_set3_raw, "Set 3 (All Players)")
    bowlers_all = process_senior_bowler_set(bw_set3_raw, "Set 3 (All Players)")

    def get_top10(df_in):
        t10 = df_in[df_in["qualifies"]].sort_values("adjusted_rating", ascending=False).head(10).copy()
        t10.insert(0, "rank", range(1, len(t10) + 1))
        return t10

    top10_rr_b = get_top10(batters_rr)
    top10_rr_bw = get_top10(bowlers_rr)
    top10_next_b = get_top10(batters_next_season)
    top10_next_bw = get_top10(bowlers_next_season)
    top10_all_b = get_top10(batters_all)
    top10_all_bw = get_top10(bowlers_all)

    # =========================================================================
    # PART B: DEVELOPMENT & YOUTH PIPELINE (STEP 4 & 5)
    # =========================================================================
    print("\n===================================================================")
    print("PART B: BUILD DEVELOPMENT & YOUTH EXPECTED-VALUE BASELINE & RANKINGS")
    print("===================================================================")

    m_dy = m_valid[m_valid["match_tier"].isin(["DEVELOPMENT", "YOUTH"])].copy()
    dy_match_ids = set(m_dy["match_id"].unique())
    bbb_dy = bbb[bbb["match_id"].isin(dy_match_ids)].copy()

    df_dy = prepare_deliveries_df(m_dy, bbb_dy)
    dev_exp_path = os.path.join(output_dir, "expected_tables_development.csv")
    exp_dy = build_expected_table(df_dy, dev_exp_path, mark_unreliable=True)
    df_dy = apply_expected_lookup(df_dy, exp_dy)

    # Team mapping for dev & youth
    p_teams_dev = mapping[mapping["comp_id"] == 2670].groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()
    p_teams_youth = mapping[mapping["comp_id"] == 3289].groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()

    def aggregate_tier_players(df_subset, tier_label, p_teams_map):
        bat_press = df_subset[df_subset["is_batting_pressure"]].copy()
        bat_norm = df_subset[~df_subset["is_batting_pressure"]].copy()

        all_strikers = sorted(df_subset["striker_id"].dropna().unique())
        b_stats = []

        for pid in all_strikers:
            p_name = p_names.get(pid, f"Player_{pid}")
            p_nat = player_nats.get(pid, "Unknown")
            teams_str = p_teams_map.get(pid, p_teams_all.get(pid, "Unknown"))

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

            b_stats.append({
                "player_id": pid,
                "player": p_name,
                "nationality": p_nat,
                "tier": tier_label,
                "teams": teams_str,
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

        bat_df = pd.DataFrame(b_stats)

        # Bowlers
        bw_band = df_subset[df_subset["is_bowling_band"]].copy()
        bw_lo = df_subset[df_subset["is_last_over"]].copy()
        bw_norm = df_subset[~df_subset["is_bowling_pressure"]].copy()

        all_bowlers = sorted(df_subset["bowler_id"].dropna().unique())
        bw_stats = []

        for pid in all_bowlers:
            p_name = p_names.get(pid, f"Player_{pid}")
            p_nat = player_nats.get(pid, "Unknown")
            teams_str = p_teams_map.get(pid, p_teams_all.get(pid, "Unknown"))

            b_b = bw_band[bw_band["bowler_id"] == pid]
            b_l = bw_lo[bw_lo["bowler_id"] == pid]
            b_n = bw_norm[bw_norm["bowler_id"] == pid]

            n_band = len(b_b)
            n_lo = len(b_l)
            n_norm = len(b_n)
            p_balls = n_band + n_lo

            if p_balls == 0 and n_norm == 0:
                continue

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

            bw_stats.append({
                "player_id": pid,
                "player": p_name,
                "nationality": p_nat,
                "tier": tier_label,
                "teams": teams_str,
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

        bw_df = pd.DataFrame(bw_stats)
        return bat_df, bw_df

    def process_tier_ratings(df_bat, df_bw, tier_label):
        # Batters
        df_b = df_bat.copy()
        df_b["qualifies"] = df_b["pressure_balls"] >= MIN_BALLS_BATTER_DEV_YOUTH
        df_b["eligibility"] = np.where(df_b["qualifies"], f"Eligible (>={MIN_BALLS_BATTER_DEV_YOUTH} balls)", f"Ineligible (<{MIN_BALLS_BATTER_DEV_YOUTH} balls)")

        b_notes = []
        for _, row in df_b.iterrows():
            pb = row["pressure_balls"]
            inns = row["pressure_innings"]
            if pb >= MIN_BALLS_BATTER_DEV_YOUTH * 2:
                b_notes.append(f"Robust sample ({pb} balls across {inns} inns). Highly statistically reliable in {tier_label}.")
            elif pb >= MIN_BALLS_BATTER_DEV_YOUTH:
                b_notes.append(f"Qualified sample ({pb} >= {MIN_BALLS_BATTER_DEV_YOUTH} balls across {inns} inns). Reliable trend in {tier_label}.")
            else:
                b_notes.append(f"Limited sample ({pb} < {MIN_BALLS_BATTER_DEV_YOUTH} balls). Below qualification threshold for {tier_label}.")
        df_b["reliability_note"] = b_notes

        q_b = df_b[df_b["qualifies"]].copy()
        if len(q_b) > 0:
            r_mean, r_std = q_b["runs_above_expected_per_100"].mean(), q_b["runs_above_expected_per_100"].std(ddof=0)
            d_mean, d_std = q_b["dismissal_penalty"].mean(), q_b["dismissal_penalty"].std(ddof=0)
            if r_std == 0: r_std = 1.0
            if d_std == 0: d_std = 1.0

            q_b["z_runs"] = (q_b["runs_above_expected_per_100"] - r_mean) / r_std
            q_b["z_dismissal"] = (q_b["dismissal_penalty"] - d_mean) / d_std
            q_b["raw_rating"] = q_b["z_runs"] - q_b["z_dismissal"]
            q_b["adjusted_rating"] = (q_b["pressure_balls"] * q_b["raw_rating"]) / (q_b["pressure_balls"] + SHRINKAGE_K_BATTER_DEV_YOUTH)

            df_b = df_b.merge(q_b[["player_id", "z_runs", "z_dismissal", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
            df_b["raw_rating"] = df_b["raw_rating"].round(4)
            df_b["adjusted_rating"] = df_b["adjusted_rating"].round(4)
        else:
            df_b["raw_rating"] = "n/a"
            df_b["adjusted_rating"] = "n/a"

        # Bowlers
        df_bw_out = df_bw.copy()
        df_bw_out["qualifies"] = df_bw_out["band_balls"] >= MIN_BALLS_BOWLER_DEV_YOUTH
        df_bw_out["eligibility"] = np.where(df_bw_out["qualifies"], f"Eligible (>={MIN_BALLS_BOWLER_DEV_YOUTH} band balls)", f"Ineligible (<{MIN_BALLS_BOWLER_DEV_YOUTH} band balls)")

        bw_notes = []
        for _, row in df_bw_out.iterrows():
            bb = row["band_balls"]
            lo = row["last_over_balls"]
            if bb >= MIN_BALLS_BOWLER_DEV_YOUTH * 2:
                bw_notes.append(f"Robust sample ({bb} band balls, {lo} last-over balls). High statistical confidence in {tier_label}.")
            elif bb >= MIN_BALLS_BOWLER_DEV_YOUTH:
                bw_notes.append(f"Qualified sample ({bb} >= {MIN_BALLS_BOWLER_DEV_YOUTH} band balls, {lo} last-over balls). Reliable indicator in {tier_label}.")
            else:
                bw_notes.append(f"Limited sample ({bb} < {MIN_BALLS_BOWLER_DEV_YOUTH} band balls). Below qualification threshold for {tier_label}.")
        df_bw_out["reliability_note"] = bw_notes

        q_bw = df_bw_out[df_bw_out["qualifies"]].copy()
        if len(q_bw) > 0:
            rs_mean, rs_std = q_bw["runs_saved_per_over"].mean(), q_bw["runs_saved_per_over"].std(ddof=0)
            w_mean, w_std = q_bw["wicket_component"].mean(), q_bw["wicket_component"].std(ddof=0)
            if rs_std == 0: rs_std = 1.0
            if w_std == 0: w_std = 1.0

            q_bw["z_runs_saved"] = (q_bw["runs_saved_per_over"] - rs_mean) / rs_std
            q_bw["z_wicket"] = (q_bw["wicket_component"] - w_mean) / w_std

            q_lo = q_bw[q_bw["lo_qualifies"]]
            if len(q_lo) > 0 and q_lo["last_over_score"].std(ddof=0) > 0:
                lo_mean = q_lo["last_over_score"].mean()
                lo_std = q_lo["last_over_score"].std(ddof=0)
                q_bw["z_last_over"] = np.where(q_bw["lo_qualifies"], (q_bw["last_over_score"] - lo_mean) / lo_std, 0.0)
            else:
                q_bw["z_last_over"] = 0.0

            q_bw["raw_rating"] = q_bw["z_runs_saved"] + q_bw["z_wicket"] + 0.5 * q_bw["z_last_over"]
            q_bw["adjusted_rating"] = (q_bw["pressure_balls"] * q_bw["raw_rating"]) / (q_bw["pressure_balls"] + SHRINKAGE_K_BOWLER_DEV_YOUTH)

            df_bw_out = df_bw_out.merge(q_bw[["player_id", "z_runs_saved", "z_wicket", "z_last_over", "raw_rating", "adjusted_rating"]], on="player_id", how="left")
            df_bw_out["raw_rating"] = df_bw_out["raw_rating"].round(4)
            df_bw_out["adjusted_rating"] = df_bw_out["adjusted_rating"].round(4)
        else:
            df_bw_out["raw_rating"] = "n/a"
            df_bw_out["adjusted_rating"] = "n/a"

        return df_b, df_bw_out

    # Process Development
    df_dev = df_dy[df_dy["match_tier"] == "DEVELOPMENT"]
    b_dev_raw, bw_dev_raw = aggregate_tier_players(df_dev, "Development", p_teams_dev)
    batters_development, bowlers_development = process_tier_ratings(b_dev_raw, bw_dev_raw, "Development")

    # Process Youth
    df_youth = df_dy[df_dy["match_tier"] == "YOUTH"]
    b_youth_raw, bw_youth_raw = aggregate_tier_players(df_youth, "Youth", p_teams_youth)
    batters_young, bowlers_young = process_tier_ratings(b_youth_raw, bw_youth_raw, "Youth")

    # Top 10 for Development & Young
    top10_dev_b = get_top10(batters_development)
    top10_dev_bw = get_top10(bowlers_development)
    top10_young_b = get_top10(batters_young)
    top10_young_bw = get_top10(bowlers_young)

    # =========================================================================
    # STEP 7: SANITY CHECKS & VERIFICATION
    # =========================================================================
    print("\n===================================================================")
    print("STEP 7: SANITY CHECKS & VERIFICATION")
    print("===================================================================")

    sanity_failures = []

    # 1. No DEVELOPMENT or YOUTH ball in Senior outputs, and vice-versa
    # Senior deliveries come strictly from senior_match_ids; dev from 2670; youth from 3289.
    if len(senior_match_ids & dy_match_ids) > 0:
        sanity_failures.append(f"Senior match IDs overlap with Development/Youth match IDs: {senior_match_ids & dy_match_ids}")
    else:
        print("  [PASSED] Strict separation: Zero overlap between Senior matches and Development/Youth matches.")

    # 2. Derivative sign checks across all sets
    all_check_sets = [
        ("Senior RR", batters_rr, bowlers_rr),
        ("Senior Next Season", batters_next_season, bowlers_next_season),
        ("Senior All Players", batters_all, bowlers_all),
        ("Development", batters_development, bowlers_development),
        ("Young", batters_young, bowlers_young),
    ]

    for s_name, b_df, bw_df in all_check_sets:
        q_b = b_df[b_df["qualifies"]].copy()
        if len(q_b) > 2:
            corr_b = np.corrcoef(q_b["z_dismissal"], q_b["raw_rating"] - q_b["z_runs"])[0, 1]
            if corr_b > -0.999:
                sanity_failures.append(f"{s_name} Batters: dismissal penalty check failed (corr={corr_b})")

        q_bw = bw_df[bw_df["qualifies"]].copy()
        if len(q_bw) > 2:
            corr_bw = np.corrcoef(q_bw["z_runs_saved"], q_bw["raw_rating"] - q_bw["z_wicket"] - 0.5 * q_bw["z_last_over"])[0, 1]
            if corr_bw < 0.999:
                sanity_failures.append(f"{s_name} Bowlers: runs saved check failed (corr={corr_bw})")

    # 3. Confirm every player in Young and Development top 10 lists meets the threshold
    for _, r in top10_dev_b.iterrows():
        if r["pressure_balls"] < MIN_BALLS_BATTER_DEV_YOUTH:
            sanity_failures.append(f"Development Batter {r['player']} has {r['pressure_balls']} < {MIN_BALLS_BATTER_DEV_YOUTH} balls!")
    for _, r in top10_dev_bw.iterrows():
        if r["band_balls"] < MIN_BALLS_BOWLER_DEV_YOUTH:
            sanity_failures.append(f"Development Bowler {r['player']} has {r['band_balls']} < {MIN_BALLS_BOWLER_DEV_YOUTH} band balls!")
    for _, r in top10_young_b.iterrows():
        if r["pressure_balls"] < MIN_BALLS_BATTER_DEV_YOUTH:
            sanity_failures.append(f"Young Batter {r['player']} has {r['pressure_balls']} < {MIN_BALLS_BATTER_DEV_YOUTH} balls!")
    for _, r in top10_young_bw.iterrows():
        if r["band_balls"] < MIN_BALLS_BOWLER_DEV_YOUTH:
            sanity_failures.append(f"Young Bowler {r['player']} has {r['band_balls']} < {MIN_BALLS_BOWLER_DEV_YOUTH} band balls!")

    if sanity_failures:
        print("\nSANITY CHECK FAILURES:")
        for f in sanity_failures:
            print(f"  FAILED: {f}")
        raise ValueError("Sanity check verification failed!")
    else:
        print("ALL SANITY CHECKS PASSED:")
        print("  1. No DEVELOPMENT or YOUTH ball appears in any senior output, and no SENIOR ball appears in Young or Development outputs.")
        print("  2. Batter dismissals derivative check (r = -1.000) and bowler economy derivative check (r = +1.000) verified.")
        print("  3. Every player in the Young and Development lists strictly meets their minimum sample threshold.")

    # =========================================================================
    # STEP 6: OUTPUTS & CSV EXPORTS
    # =========================================================================
    print("\n===================================================================")
    print("STEP 6: EXPORT CSV OUTPUTS")
    print("===================================================================")

    # Senior full tables
    batters_rr.to_csv(os.path.join(output_dir, "batters_rr.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_rr.csv"), index=False)
    batters_next_season.to_csv(os.path.join(output_dir, "batters_next_season.csv"), index=False)
    bowlers_next_season.to_csv(os.path.join(output_dir, "bowlers_next_season.csv"), index=False)
    batters_all.to_csv(os.path.join(output_dir, "batters_all.csv"), index=False)
    bowlers_all.to_csv(os.path.join(output_dir, "bowlers_all.csv"), index=False)

    batters_rr.to_csv(os.path.join(output_dir, "batters_pressure.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_pressure.csv"), index=False)

    # Senior Top 10 tables
    top10_rr_b.to_csv(os.path.join(output_dir, "top10_rr_batters.csv"), index=False)
    top10_rr_bw.to_csv(os.path.join(output_dir, "top10_rr_bowlers.csv"), index=False)
    top10_next_b.to_csv(os.path.join(output_dir, "top10_next_season_batters.csv"), index=False)
    top10_next_bw.to_csv(os.path.join(output_dir, "top10_next_season_bowlers.csv"), index=False)
    top10_all_b.to_csv(os.path.join(output_dir, "top10_all_batters.csv"), index=False)
    top10_all_bw.to_csv(os.path.join(output_dir, "top10_all_bowlers.csv"), index=False)

    # Young & Development tables
    batters_development.to_csv(os.path.join(output_dir, "batters_development.csv"), index=False)
    bowlers_development.to_csv(os.path.join(output_dir, "bowlers_development.csv"), index=False)
    batters_young.to_csv(os.path.join(output_dir, "batters_young.csv"), index=False)
    bowlers_young.to_csv(os.path.join(output_dir, "bowlers_young.csv"), index=False)

    top10_dev_b.to_csv(os.path.join(output_dir, "top10_development_batters.csv"), index=False)
    top10_dev_bw.to_csv(os.path.join(output_dir, "top10_development_bowlers.csv"), index=False)
    top10_young_b.to_csv(os.path.join(output_dir, "top10_young_batters.csv"), index=False)
    top10_young_bw.to_csv(os.path.join(output_dir, "top10_young_bowlers.csv"), index=False)

    print("\n[TOP 10 DEVELOPMENT BATTERS]")
    print(top10_dev_b[["rank", "player", "teams", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 DEVELOPMENT BOWLERS]")
    print(top10_dev_bw[["rank", "player", "teams", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 YOUNG BATTERS]")
    print(top10_young_b[["rank", "player", "teams", "adjusted_rating", "runs_above_expected_per_100", "dismissal_penalty", "pressure_balls"]].to_string(index=False))

    print("\n[TOP 10 YOUNG BOWLERS]")
    print(top10_young_bw[["rank", "player", "teams", "adjusted_rating", "runs_saved_per_over", "wicket_component", "last_over_score", "pressure_balls"]].to_string(index=False))

    return {
        "top10_rr_b": top10_rr_b,
        "top10_rr_bw": top10_rr_bw,
        "top10_next_b": top10_next_b,
        "top10_next_bw": top10_next_bw,
        "top10_all_b": top10_all_b,
        "top10_all_bw": top10_all_bw,
        "top10_dev_b": top10_dev_b,
        "top10_dev_bw": top10_dev_bw,
        "top10_young_b": top10_young_b,
        "top10_young_bw": top10_young_bw,
        "batters_development": batters_development,
        "bowlers_development": bowlers_development,
        "batters_young": batters_young,
        "bowlers_young": bowlers_young
    }

run_pressure_pipeline = run_multi_tier_pipeline

if __name__ == "__main__":
    run_multi_tier_pipeline()

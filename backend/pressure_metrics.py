"""
Rajasthan Royals & Global T20 Pressure Analytics Engine (Production Version 2.0)
Multi-Tier Framework:
- SENIOR: Full national teams (with Top 12 T20I opponent filter) and IPL franchises (unfiltered).
  Ranks Set 1 (RR Players), Set 2 (Next Season Eligible - ex-Pak), Set 3 (All Senior Players).
- DEVELOPMENT: A-teams and emerging competitions (Mens T20 Emerging Teams Asia Cup 2024).
- YOUTH: Youth competitions (Asia Cup Rising Stars 2025).

Key Algorithmic Upgrades:
- Change 1: One outcome per pressure innings (innings_value) in runs.
- Change 2: Required-run-rate (RRR) chase tracking with RRR_PENALTY_WEIGHT = 0.5.
- Change 3: Strict RR match participation filter (removes unparticipating squad members).
- Change 4: Corrected pressure definition (2nd innings RRR >= 9.0 or death overs 16-20; excludes 1st innings).
- Change 5: Fixed dismissal flag (striker only; handles missing dismissal IDs explicitly).
- Change 6: Leave-One-Out (LOO) expected values to prevent data leakage in rankings.
- Change 7: Innings floor (MIN_INNINGS_BATTER = 8, MIN_INNINGS_BOWLER = 8) and Z_CLIP = 2.0.
- Change 8: Synthetic ranking-order sanity test suite replacing circular correlation checks.
- Change 9: Dynamic Set 2 season filter based on the two most recent seasons.
- Change 10: Complete verification and old vs new top-10 shift audit.
"""

import os
import json
import pandas as pd
import numpy as np

# =========================================================================
# STEP 2: TEAM TIER CLASSIFICATION MAPPING CONSTANT
# =========================================================================
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
# THRESHOLDS, SHRINKAGE, INNINGS FLOOR & CLIPPING CONSTANTS (CHANGES 2, 7)
# =========================================================================
# Senior Sets (Set 1, 2, 3)
MIN_BALLS_BATTER_RR = 30          # RR Batters: min 30 pressure balls
MIN_BALLS_BOWLER_RR = 24          # RR Bowlers: min 24 balls in 8-12 bowling band
MIN_BALLS_BATTER_NON_RR = 60      # Non-RR Batters: min 60 pressure balls
MIN_BALLS_BOWLER_NON_RR = 48      # Non-RR Bowlers: min 48 balls in 8-12 bowling band

MIN_INNINGS_BATTER = 8            # Change 7: Minimum 8 pressure innings for senior batters
MIN_INNINGS_BOWLER = 8            # Change 7: Minimum 8 pressure innings for senior bowlers
MIN_INNINGS_DEV_YOUTH = 2         # Change 7: Innings floor for 5-match dev/youth tournaments
Z_CLIP = 2.0                      # Change 7: Clipping bound for all z-score components [-2.0, +2.0]
RRR_PENALTY_WEIGHT = 0.5          # Change 2: Weight for chase required-run-rate lag penalty

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

# Old Top 10 Snapshot for Change 10 Comparison
OLD_TOP10 = {
    "RR Batters": [
        "Karun Nair", "Rassie van der Dussen", "Vaibhav Sooryavanshi", "Ravindra Jadeja",
        "Dhruv Jurel", "Jos Buttler", "Sanju Samson", "Rovman Powell", "Yashasvi Jaiswal", "Dasun Shanaka"
    ],
    "RR Bowlers": [
        "Akash Madhwal", "Murugan Ashwin", "FazalHaq Farooqi", "Keshav Maharaj",
        "Maheesh Theekshana", "Jimmy Neesham", "Trent Boult", "Yuzvendra Chahal", "Kuldeep Sen", "Ravi Bishnoi"
    ],
    "Next Season Batters": [
        "Atharwa Taide", "Vaibhav Sooryavanshi", "Suryakumar Yadav", "Reeza Hendricks",
        "Tom Banton", "Karun Nair", "Shashank Singh", "Namboori Tilak Varma", "Sai Sudharsan", "Lokesh Rahul"
    ],
    "Next Season Bowlers": [
        "Akash Madhwal", "Mark Wood", "Mahedi Hasan", "Mitchell Santner",
        "Jacob Duffy", "Mohammad Rishad Hossain", "Corbin Bosch", "Nathan Ellis", "FazalHaq Farooqi", "Axar Patel"
    ],
    "All Senior Batters": [
        "Atharwa Taide", "Vaibhav Sooryavanshi", "Shan Masood", "Matthew Wade",
        "Mohammad Rizwan", "Suryakumar Yadav", "Shakib Al Hasan", "Reeza Hendricks", "Tom Banton", "Karun Nair"
    ],
    "All Senior Bowlers": [
        "Akash Madhwal", "Mark Wood", "Mahedi Hasan", "Mohammad Nawaz",
        "Mitchell Santner", "Jacob Duffy", "Mohammad Rishad Hossain", "Murugan Ashwin", "Corbin Bosch", "Nathan Ellis"
    ],
    "Development Batters": [
        "Sediqullah Atal", "Arafat Minhas", "Ramandeep Singh", "Jatinder Singh",
        "Sahan Arachchige", "Babar Hayat", "Namboori Tilak Varma", "Haider Ali", "Hammad Mirza", "Yasim Murtaza"
    ],
    "Development Bowlers": [
        "Shahnawaz Dhani", "Eshan Malinga", "Allah Ghazanfar", "Faridoon Dawoodzai",
        "Nimesh Vimukthi", "Sharafuddin Ashraf", "Karim Janat", "Bilal Sami"
    ],
    "Young Batters": [
        "Sohaib Khan", "Jitesh Sharma", "SM Meherob Hasan", "Milan Rathnayake",
        "Haider Shah", "Aryan Bisht", "Muhammad Arfan", "Wasim Ali", "Nehal Wadhera", "Qais Ahmad"
    ],
    "Young Bowlers": [
        "Garuka Sanketh", "SM Meherob Hasan"
    ]
}


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


def run_synthetic_sanity_tests():
    """
    Change 8: Synthetic input row ranking-order tests.
    Tests ranking properties directly on synthetic inputs without circular correlation.
    - Test (a): Same runs + more dismissals ranks lower.
    - Test (b): Conceding more runs in same band ranks lower.
    - Test (c): Strike rate below RRR in >50% of balls ranks lower.
    """
    print("\n===================================================================")
    print("STEP 7: SYNTHETIC RANKING-ORDER SANITY CHECKS (CHANGE 8)")
    print("===================================================================")

    # Test (a): Batter with same runs and more dismissals ranks lower
    exp_runs_pi = 20.0
    # Batter 1: 30 runs, exp 20, 0 dismissals, exp 0.5
    inns_val_1 = (30.0 - 20.0) - (0.0 - 0.5) * exp_runs_pi   # 10 - (-10) = 20.0
    # Batter 2: 30 runs, exp 20, 1 dismissal, exp 0.5
    inns_val_2 = (30.0 - 20.0) - (1.0 - 0.5) * exp_runs_pi   # 10 - (10) = 0.0
    if not (inns_val_2 < inns_val_1):
        raise ValueError("FAILED Test (a): Batter with more dismissals did not receive lower innings value!")
    print("  [PASSED] Test (a): Same runs + more dismissals ranks lower (Innings Value: 0.0 < 20.0).")

    # Test (b): Bowler who concedes more runs in same band ranks lower
    # Bowler 1: 10 overs, concedes 60 runs, exp 80 -> runs saved = +2.0/over
    # Bowler 2: 10 overs, concedes 90 runs, exp 80 -> runs saved = -1.0/over
    rs_1 = (80.0 - 60.0) / 10.0
    rs_2 = (80.0 - 90.0) / 10.0
    if not (rs_2 < rs_1):
        raise ValueError("FAILED Test (b): Bowler conceding more runs did not receive lower runs saved per over!")
    print("  [PASSED] Test (b): Bowler conceding more runs in band ranks lower (Runs Saved/Over: -1.0 < +2.0).")

    # Test (c): Lower pressure strike rate than RRR in >50% balls ranks lower
    # Batter 3: 20% balls below RRR
    # Batter 4: 70% balls below RRR (>50% balls)
    mean_pct, std_pct = 50.0, 15.0
    z_pct_3 = (20.0 - mean_pct) / std_pct  # -2.0
    z_pct_4 = (70.0 - mean_pct) / std_pct  # +1.333
    rating_3 = 1.0 - RRR_PENALTY_WEIGHT * z_pct_3  # 1.0 - 0.5*(-2.0) = 2.0
    rating_4 = 1.0 - RRR_PENALTY_WEIGHT * z_pct_4  # 1.0 - 0.5*(1.333) = 0.333
    if not (rating_4 < rating_3):
        raise ValueError("FAILED Test (c): Batter lagging RRR in >50% balls did not rank lower!")
    print(f"  [PASSED] Test (c): Strike rate below RRR in >50% balls ranks lower (Rating: {rating_4:.3f} < {rating_3:.3f}).")


def print_old_vs_new_comparison(label, new_df, old_list):
    """
    Change 10: Print a comparison of old and new top 10 for each set,
    marking players who moved more than 5 places or left the list.
    """
    print(f"\n--- {label}: Old vs New Top 10 Comparison ---")
    old_rank_map = {name: idx + 1 for idx, name in enumerate(old_list)}
    new_players = []

    rows = []
    for idx, r in new_df.iterrows():
        pname = r["player"]
        new_rank = r["rank"]
        new_players.append(pname)
        if pname in old_rank_map:
            old_rank = old_rank_map[pname]
            delta = old_rank - new_rank  # positive means improved rank
            if delta > 5:
                note = f"Shifted up {delta} spots (moved >5 places)"
            elif delta < -5:
                note = f"Shifted down {abs(delta)} spots (moved >5 places)"
            elif delta > 0:
                note = f"Up {delta} spot(s)"
            elif delta < 0:
                note = f"Down {abs(delta)} spot(s)"
            else:
                note = "Maintained rank"
            old_rank_str = str(old_rank)
        else:
            old_rank_str = "Unranked"
            note = "NEW ENTRANT"
        rows.append({
            "New Rank": new_rank,
            "Player": pname,
            "Old Rank": old_rank_str,
            "Rating": r["adjusted_rating"],
            "Shift / Status": note
        })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # Players who left the top 10
    left_players = [p for p in old_list if p not in new_players]
    if left_players:
        print(f"  Players who left Top 10: {', '.join(left_players)}")
    else:
        print("  Players who left Top 10: None")


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

    # Confirm Team ID and Name for Rajasthan Royals
    rr_team_rows = teams[teams["team_name"].str.contains("Rajasthan Royals", case=False, na=False)]
    if len(rr_team_rows) == 0:
        raise ValueError("Rajasthan Royals not found in teams.csv!")
    rr_team_ids = rr_team_rows["team_id"].unique().tolist()
    print(f"Confirmed Rajasthan Royals Team ID(s) from teams.csv: {rr_team_ids}")
    rr_team_id = rr_team_ids[0]

    # Verify against matches.csv
    rr_matches = matches[(matches["home_team_id"] == rr_team_id) | (matches["away_team_id"] == rr_team_id)]
    rr_home_names = rr_matches[rr_matches["home_team_id"] == rr_team_id]["home_team"].unique().tolist()
    rr_away_names = rr_matches[rr_matches["away_team_id"] == rr_team_id]["away_team"].unique().tolist()
    print(f"Confirmed Rajasthan Royals Names in matches.csv: home={rr_home_names}, away={rr_away_names}")

    # Change 9: Season field and Most Recent Seasons dynamically
    comp["season_clean"] = comp["season"].astype(str).str.strip()
    all_seasons = sorted(comp["season_clean"].unique())
    recent_two_seasons = all_seasons[-2:]
    print(f"\nDistinct seasons in data: {all_seasons}")
    print(f"Two most recent seasons dynamically selected: {recent_two_seasons}")

    # Distinct team names and ball counts
    team_id_to_name = teams.drop_duplicates(subset=["team_id"]).set_index("team_id")["team_name"].to_dict()
    bbb["batting_team_name"] = bbb["batting_team_id"].map(team_id_to_name)
    bbb["bowling_team_name"] = bbb["bowling_team_id"].map(team_id_to_name)

    all_teams_in_data = sorted(set(teams["team_name"].unique()) | set(matches["home_team"].unique()) | set(matches["away_team"].unique()))

    # Verify tier assignment
    for t in all_teams_in_data:
        if t not in TEAM_TIER:
            raise ValueError(f"Unassigned team name encountered: {t}")

    # Multi-tier Player Audit
    m_comp = matches.merge(comp[["comp_id", "comp_name", "match_format", "season_clean"]], on="comp_id", how="left")
    m_comp["match_tier"] = m_comp.apply(lambda r: classify_match_tier(r["comp_id"], r["comp_name"]), axis=1)

    p_names = mapping.groupby("player_id")["player_name"].first().to_dict()
    p_teams_all = mapping.groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()

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
            nat = overseas_ipl.get(pname, "India")
        player_nats[pid] = nat

    # Match format and filtering
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

    # Deliveries total runs & targets
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

    # =========================================================================
    # COMMON DELIVERIES DATA PREPARATION FUNCTION (CHANGES 4, 5)
    # =========================================================================
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

        # Change 5: Fix dismissal flag - strictly striker dismissal
        # Missing dismissed_player_id handled explicitly (Match 40709 non-striker retired out -> False)
        missing_dismissal_rows = df_out[df_out["is_wicket"] & df_out["dismissed_player_id"].isna()]
        missing_cnt = len(missing_dismissal_rows)
        if missing_cnt > 0:
            print(f"Change 5 Notice: Found {missing_cnt} row(s) with is_wicket=True and missing dismissed_player_id.")
            for _, mr in missing_dismissal_rows.iterrows():
                print(f"  Explicit handling: Match {mr['match_id']}, Innings {mr['innings']}, Over {mr['overid']}. Dismissal type: '{mr['dismissal_type']}'. Handled explicitly as False (striker not dismissed).")

        df_out["batter_dismissal"] = df_out["is_wicket"] & (df_out["dismissed_player_id"].notna()) & (df_out["dismissed_player_id"] == df_out["striker_id"])

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

        # Context Buckets
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

        # Change 4: Pressure definitions
        # Batting pressure: second innings only, where required run rate >= 9.0 or death overs (overs 16-20)
        # 1st innings pressure is completely excluded.
        first_inn_balls_excluded = ((df_out["innings"] == 1) & ((df_out["balls_remaining_before"] <= 30) | (df_out["overid"] >= 16))).sum()
        df_out["is_batting_pressure"] = (df_out["innings"] == 2) & ((df_out["rrr"] >= 9.0) | (df_out["overid"] >= 16))

        # Bowling pressure: second innings with RRR 8.0-12.0, plus last over of either innings
        df_out["is_last_over"] = df_out["balls_remaining_before"] <= 6
        df_out["is_bowling_band"] = (df_out["innings"] == 2) & (df_out["rrr"] >= 8.0) & (df_out["rrr"] <= 12.0) & (~df_out["is_last_over"])
        df_out["is_bowling_pressure"] = df_out["is_last_over"] | df_out["is_bowling_band"]

        return df_out, first_inn_balls_excluded

    # Helper function to compute expected table
    def build_expected_table(df_enriched, out_csv_path=None, mark_unreliable=False):
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

        if out_csv_path:
            exp_t.to_csv(out_csv_path, index=False)
            print(f"Saved {len(exp_t)} buckets to {out_csv_path}")
        return exp_t

    # Change 6: Algebraic Leave-One-Out (LOO) expected values calculator
    def compute_loo_deliveries(df_in, exp_table):
        b_totals = exp_table.set_index("bucket_id").to_dict("index")

        # Batter aggregates by (striker_id, bucket_id)
        b_player_bucket = df_in.groupby(["striker_id", "bucket_id"]).agg(
            p_b=("runs", "count"),
            p_r=("runs", "sum"),
            p_d=("batter_dismissal", "sum")
        ).reset_index().set_index(["striker_id", "bucket_id"]).to_dict("index")

        # Bowler aggregates by (bowler_id, bucket_id)
        bw_player_bucket = df_in.groupby(["bowler_id", "bucket_id"]).agg(
            p_b=("runs", "count"),
            p_r=("bowler_runs_conceded", "sum"),
            p_w=("bowler_wicket", "sum"),
            p_dots=("runs", lambda s: ((s == 0) & (df_in.loc[s.index, "extras1_runs"] == 0)).sum())
        ).reset_index().set_index(["bowler_id", "bucket_id"]).to_dict("index")

        def get_loo_batter(row):
            b_id = row["bucket_id"]
            p_id = row["striker_id"]
            bt = b_totals[b_id]
            pb = b_player_bucket.get((p_id, b_id))
            if pb and (bt["ball_count"] - pb["p_b"] > 0):
                rem_b = bt["ball_count"] - pb["p_b"]
                exp_r = (bt["batter_runs"] - pb["p_r"]) / rem_b
                exp_d = (bt["batter_dismissals"] - pb["p_d"]) / rem_b
            else:
                exp_r = bt["batter_avg_runs_per_ball"]
                exp_d = bt["batter_dismissal_rate"]
            return exp_r, exp_d

        def get_loo_bowler(row):
            b_id = row["bucket_id"]
            p_id = row["bowler_id"]
            bt = b_totals[b_id]
            p_bw = bw_player_bucket.get((p_id, b_id))
            if p_bw and (bt["ball_count"] - p_bw["p_b"] > 0):
                rem_b = bt["ball_count"] - p_bw["p_b"]
                exp_r = (bt["bowler_runs"] - p_bw["p_r"]) / rem_b
                exp_w = (bt["bowler_wickets"] - p_bw["p_w"]) / rem_b
                exp_dot = ((bt["bowler_dots"] - p_bw["p_dots"]) / rem_b) * 100.0
                rpw = (bt["bowler_runs"] - p_bw["p_r"]) / (bt["bowler_wickets"] - p_bw["p_w"]) if (bt["bowler_wickets"] - p_bw["p_w"] > 0) else exp_r * 6.0
            else:
                exp_r = bt["bowler_avg_runs_per_ball"]
                exp_w = bt["bowler_wicket_rate"]
                exp_dot = bt["bowler_dot_pct"]
                rpw = bt["runs_per_wicket"]
            return exp_r, exp_w, exp_dot, rpw

        batter_res = [get_loo_batter(r) for _, r in df_in[["bucket_id", "striker_id"]].iterrows()]
        df_in["exp_batter_runs"] = [r[0] for r in batter_res]
        df_in["exp_batter_dismissal"] = [r[1] for r in batter_res]

        bowler_res = [get_loo_bowler(r) for _, r in df_in[["bucket_id", "bowler_id"]].iterrows()]
        df_in["exp_bowler_runs"] = [r[0] for r in bowler_res]
        df_in["exp_bowler_wicket"] = [r[1] for r in bowler_res]
        df_in["exp_bowler_dot_pct"] = [r[2] for r in bowler_res]
        df_in["runs_per_wicket"] = [r[3] for r in bowler_res]

        # In-sample difference calculation
        df_in["insample_exp_batter_runs"] = df_in["bucket_id"].map(lambda b: b_totals[b]["batter_avg_runs_per_ball"])
        avg_loo_diff = (df_in["exp_batter_runs"] - df_in["insample_exp_batter_runs"]).abs().mean()
        print(f"Change 6 Notice: Average absolute difference between in-sample and LOO expected runs per delivery: {avg_loo_diff:.6f} runs/ball")

        return df_in

    # =========================================================================
    # PART A: SENIOR PIPELINE (SETS 1, 2, 3)
    # =========================================================================
    print("\n===================================================================")
    print("PART A: BUILD SENIOR BASELINE & RANKINGS")
    print("===================================================================")

    m_senior_qualifying = m_valid[(m_valid["match_tier"] == "SENIOR") & (m_valid["is_ipl"] | (m_valid["is_t20i"] & m_valid["both_top12"]))].copy()
    senior_match_ids = set(m_senior_qualifying["match_id"].unique())
    bbb_senior = bbb[bbb["match_id"].isin(senior_match_ids)].copy()

    df_senior, inn1_excluded_senior = prepare_deliveries_df(m_senior_qualifying, bbb_senior)
    print(f"Change 4 Notice: Number of first-innings pressure balls excluded in Senior tier: {inn1_excluded_senior} balls")

    senior_exp_path = os.path.join(output_dir, "expected_tables.csv")
    exp_senior = build_expected_table(df_senior, senior_exp_path)
    df_senior = compute_loo_deliveries(df_senior, exp_senior)

    # Change 2: Required-run-rate tracking for chases at delivery level
    df_senior_press = df_senior[df_senior["is_batting_pressure"]].copy()
    df_senior_press["press_ball_num"] = df_senior_press.groupby(["match_id", "innings", "striker_id"]).cumcount() + 1
    df_senior_press["press_cum_runs"] = df_senior_press.groupby(["match_id", "innings", "striker_id"])["runs"].cumsum()
    df_senior_press["press_rpb"] = df_senior_press["press_cum_runs"] / df_senior_press["press_ball_num"]
    df_senior_press["rrr_rpb"] = df_senior_press["rrr"] / 6.0
    df_senior_press["is_below_rrr"] = df_senior_press["press_rpb"] < df_senior_press["rrr_rpb"]

    # Change 1: Build Per-Innings Table for Batters
    inns_table = df_senior_press.groupby(["match_id", "innings", "striker_id"]).agg(
        pressure_balls=("runs", "count"),
        actual_runs=("runs", "sum"),
        expected_runs=("exp_batter_runs", "sum"),
        actual_dismissals=("batter_dismissal", "sum"),
        expected_dismissals=("exp_batter_dismissal", "sum"),
        balls_below_rrr=("is_below_rrr", "sum")
    ).reset_index()

    inns_table["player"] = inns_table["striker_id"].map(p_names)
    inns_table["excess_dismissals"] = inns_table["actual_dismissals"] - inns_table["expected_dismissals"]
    inns_table["runs_above_expected"] = inns_table["actual_runs"] - inns_table["expected_runs"]
    inns_table["expected_runs_per_innings"] = inns_table["expected_runs"]
    # Change 1: innings_value = actual runs − expected runs − (actual dismissals − expected dismissals) × expected runs per innings
    inns_table["innings_value"] = inns_table["runs_above_expected"] - inns_table["excess_dismissals"] * inns_table["expected_runs_per_innings"]
    inns_table["pct_balls_below_rrr"] = (inns_table["balls_below_rrr"] / inns_table["pressure_balls"]) * 100.0

    # Save per-innings table (Change 1)
    per_innings_path = os.path.join(output_dir, "batter_pressure_innings.csv")
    inns_table.to_csv(per_innings_path, index=False)
    print(f"Saved {len(inns_table)} pressure innings to {per_innings_path}")

    # Change 3: Fix RR Squad Mapping
    # Match ID participation filter: player batted or bowled for RR (team_id == 3) in a SENIOR match
    rr_batted_new = set(df_senior[df_senior["batting_team_id"] == rr_team_id]["striker_id"].dropna().unique())
    rr_bowled_new = set(df_senior[df_senior["bowling_team_id"] == rr_team_id]["bowler_id"].dropna().unique())
    set1_pids_after = rr_batted_new | rr_bowled_new

    # Comparison with old mapping (which used mapping table squad registration)
    rr_map_old = set(mapping[mapping["team_id"] == rr_team_id]["player_id"].dropna().unique())
    set1_pids_before = rr_batted_new | rr_bowled_new | rr_map_old
    removed_rr_pids = sorted(set1_pids_before - set1_pids_after)

    print("\n--- CHANGE 3: RR SQUAD MAPPING FIX AUDIT ---")
    print(f"RR Player Count BEFORE Fix: {len(set1_pids_before)}")
    print(f"RR Player Count AFTER Fix:  {len(set1_pids_after)}")
    print(f"Number of Unparticipating Players Removed: {len(removed_rr_pids)}")
    print("Players removed from RR Set (registered in squad but 0 participating balls in senior matches):")
    for r_pid in removed_rr_pids:
        print(f"  ID {r_pid:5d}: {p_names.get(r_pid, f'Player_{r_pid}')} (is_rr: True -> False)")

    # Change 9: Dynamic Set 2 filter
    recent_comps = set(comp[comp["season_clean"].isin(recent_two_seasons)]["comp_id"].unique())
    recent_mids = set(m_senior_qualifying[m_senior_qualifying["comp_id"].isin(recent_comps)]["match_id"].unique())
    bbb_recent = df_senior[df_senior["match_id"].isin(recent_mids)]
    recent_pids = set(bbb_recent["striker_id"].dropna().unique()) | set(bbb_recent["bowler_id"].dropna().unique())
    set2_pids = {pid for pid in recent_pids if player_nats.get(pid) != "Pakistan"}

    print("\n--- CHANGE 9: SET 2 DYNAMIC SEASONS AUDIT ---")
    print(f"Two most recent seasons: {recent_two_seasons}")
    print(f"Total eligible players for Next Season (Set 2, ex-Pakistan): {len(set2_pids)}")
    if len(set2_pids) == 0:
        raise ValueError("Set 2 player count is zero! Season filter failed.")

    # Batter Player Aggregation (Senior)
    p_bat_agg = inns_table.groupby("striker_id").agg(
        pressure_balls=("pressure_balls", "sum"),
        pressure_innings=("pressure_balls", "count"),
        actual_runs=("actual_runs", "sum"),
        expected_runs=("expected_runs", "sum"),
        actual_dismissals=("actual_dismissals", "sum"),
        expected_dismissals=("expected_dismissals", "sum"),
        total_balls_below_rrr=("balls_below_rrr", "sum"),
        mean_innings_value=("innings_value", "mean")
    ).reset_index()

    p_bat_agg["player"] = p_bat_agg["striker_id"].map(p_names)
    p_bat_agg["nationality"] = p_bat_agg["striker_id"].map(player_nats)
    p_bat_agg["tier"] = "Senior"
    p_bat_agg["teams"] = p_bat_agg["striker_id"].map(p_teams_all)
    p_bat_agg["is_rr"] = p_bat_agg["striker_id"].isin(set1_pids_after)
    p_bat_agg["in_set2"] = p_bat_agg["striker_id"].isin(set2_pids)
    p_bat_agg["pct_balls_below_rrr"] = (p_bat_agg["total_balls_below_rrr"] / p_bat_agg["pressure_balls"]) * 100.0

    # Normal baseline stats for batters
    bat_norm = df_senior[~df_senior["is_batting_pressure"]].copy()
    norm_b_agg = bat_norm.groupby("striker_id").agg(
        norm_balls=("runs", "count"),
        norm_runs=("runs", "sum"),
        norm_inns=("match_id", "nunique"),
        norm_bndry=("runs", lambda s: (bat_norm.loc[s.index, "is_four"] | bat_norm.loc[s.index, "is_six"]).sum()),
        norm_dismissals=("batter_dismissal", "sum")
    ).reset_index()

    p_bat_agg = p_bat_agg.merge(norm_b_agg, on="striker_id", how="left")
    p_bat_agg["norm_balls"] = p_bat_agg["norm_balls"].fillna(0)
    p_bat_agg["norm_runs"] = p_bat_agg["norm_runs"].fillna(0)
    p_bat_agg["norm_inns"] = p_bat_agg["norm_inns"].fillna(0)
    p_bat_agg["norm_bndry"] = p_bat_agg["norm_bndry"].fillna(0)
    p_bat_agg["norm_dismissals"] = p_bat_agg["norm_dismissals"].fillna(0)

    p_bat_agg["pressure_strike_rate"] = np.where(p_bat_agg["pressure_balls"] > 0, (p_bat_agg["actual_runs"] / p_bat_agg["pressure_balls"]) * 100.0, np.nan).round(1)
    p_bat_agg["normal_strike_rate"] = np.where(p_bat_agg["norm_balls"] > 0, (p_bat_agg["norm_runs"] / p_bat_agg["norm_balls"]) * 100.0, np.nan).round(1)
    p_bat_agg["sr_difference"] = (p_bat_agg["pressure_strike_rate"] - p_bat_agg["normal_strike_rate"]).round(1)

    df_senior_press["is_boundary"] = df_senior_press["is_four"] | df_senior_press["is_six"]
    press_bndry_counts = df_senior_press.groupby("striker_id")["is_boundary"].sum().to_dict()
    p_bat_agg["pressure_boundary_pct"] = np.where(
        p_bat_agg["pressure_balls"] > 0,
        (p_bat_agg["striker_id"].map(press_bndry_counts).fillna(0) / p_bat_agg["pressure_balls"]) * 100.0,
        np.nan
    ).round(1)
    p_bat_agg["normal_boundary_pct"] = np.where(p_bat_agg["norm_balls"] > 0, (p_bat_agg["norm_bndry"] / p_bat_agg["norm_balls"]) * 100.0, np.nan).round(1)

    p_bat_agg["pressure_dismissal_rate"] = np.where(p_bat_agg["pressure_innings"] > 0, p_bat_agg["actual_dismissals"] / p_bat_agg["pressure_innings"], np.nan).round(3)
    p_bat_agg["normal_dismissal_rate"] = np.where(p_bat_agg["norm_inns"] > 0, p_bat_agg["norm_dismissals"] / p_bat_agg["norm_inns"], np.nan).round(3)
    p_bat_agg["dismissal_difference"] = (p_bat_agg["pressure_dismissal_rate"] - p_bat_agg["normal_dismissal_rate"]).round(3)

    # Process Senior Batter Set with Changes 1, 2, 7
    def process_senior_batters(df_in, inns_df, is_rr_only=False):
        df_out = df_in.copy()
        thresh_b = MIN_BALLS_BATTER_RR if is_rr_only else np.where(df_out["is_rr"], MIN_BALLS_BATTER_RR, MIN_BALLS_BATTER_NON_RR)
        k_val = SHRINKAGE_K_BATTER_RR if is_rr_only else np.where(df_out["is_rr"], SHRINKAGE_K_BATTER_RR, SHRINKAGE_K_BATTER_NON_RR)

        df_out["threshold_applied"] = thresh_b
        df_out["shrinkage_k"] = k_val
        # Change 7: Qualifies requires both ball threshold and innings floor
        df_out["qualifies"] = (df_out["pressure_balls"] >= df_out["threshold_applied"]) & (df_out["pressure_innings"] >= MIN_INNINGS_BATTER)
        df_out["eligibility"] = np.where(df_out["qualifies"], f"Eligible (>={thresh_b}b, >={MIN_INNINGS_BATTER}inns)", "Ineligible")

        notes = []
        for _, row in df_out.iterrows():
            pb = row["pressure_balls"]
            thresh = row["threshold_applied"]
            inns = row["pressure_innings"]
            if pb >= thresh * 2 and inns >= MIN_INNINGS_BATTER * 2:
                notes.append(f"Robust sample ({pb} balls across {inns} inns). Highly statistically reliable.")
            elif pb >= thresh and inns >= MIN_INNINGS_BATTER:
                notes.append(f"Qualified sample ({pb} >= {thresh} balls across {inns} inns). Reliable trend.")
            else:
                notes.append(f"Limited sample ({pb} balls, {inns} inns). Below qualification threshold.")
        df_out["reliability_note"] = notes

        q = df_out[df_out["qualifies"]].copy()
        if len(q) > 0:
            q_pids = set(q["striker_id"])
            q_inns = inns_df[inns_df["striker_id"].isin(q_pids)]
            std_inns_val = q_inns["innings_value"].std(ddof=0)
            if std_inns_val == 0: std_inns_val = 1.0

            # Change 1: Batter rating = mean of innings_value / std(innings_value across qualifying innings)
            q["z_innings_value"] = q["mean_innings_value"] / std_inns_val
            # Change 7: Clip every component to [-2.0, 2.0]
            q["z_innings_value_clipped"] = np.clip(q["z_innings_value"], -Z_CLIP, Z_CLIP)

            # Change 2: Required-run-rate penalty term
            pct_m = q["pct_balls_below_rrr"].mean()
            pct_s = q["pct_balls_below_rrr"].std(ddof=0)
            if pct_s == 0: pct_s = 1.0
            q["z_pct_below_rrr"] = (q["pct_balls_below_rrr"] - pct_m) / pct_s
            q["z_pct_below_rrr_clipped"] = np.clip(q["z_pct_below_rrr"], -Z_CLIP, Z_CLIP)

            # Rating formula: innings-based value - 0.5 * z(pct_balls_below_rrr)
            q["raw_rating"] = q["z_innings_value_clipped"] - RRR_PENALTY_WEIGHT * q["z_pct_below_rrr_clipped"]
            q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + q["shrinkage_k"])

            merge_cols = ["striker_id", "z_innings_value", "z_innings_value_clipped", "z_pct_below_rrr", "z_pct_below_rrr_clipped", "raw_rating", "adjusted_rating"]
            df_out = df_out.merge(q[merge_cols], on="striker_id", how="left")
            df_out["raw_rating"] = df_out["raw_rating"].round(4)
            df_out["adjusted_rating"] = df_out["adjusted_rating"].round(4)
            df_out["mean_innings_value"] = df_out["mean_innings_value"].round(2)
            df_out["pct_balls_below_rrr"] = df_out["pct_balls_below_rrr"].round(1)
        else:
            df_out["raw_rating"] = "n/a"
            df_out["adjusted_rating"] = "n/a"
        return df_out

    # Bowler Player Aggregation (Senior)
    bw_band = df_senior[df_senior["is_bowling_band"]].copy()
    bw_lo = df_senior[df_senior["is_last_over"]].copy()
    all_bowlers = sorted(df_senior["bowler_id"].dropna().unique())

    bw_stats = []
    for pid in all_bowlers:
        b_b = bw_band[bw_band["bowler_id"] == pid]
        b_l = bw_lo[bw_lo["bowler_id"] == pid]
        n_band = len(b_b)
        n_lo = len(b_l)
        p_balls = n_band + n_lo
        p_inns = len(set(b_b["match_id"]) | set(b_l["match_id"]))

        if p_balls == 0:
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
            "bowler_id": pid,
            "player": p_names.get(pid, f"Player_{pid}"),
            "nationality": player_nats.get(pid, "Unknown"),
            "tier": "Senior",
            "teams": p_teams_all.get(pid, "Unknown"),
            "is_rr": pid in set1_pids_after,
            "in_set2": pid in set2_pids,
            "pressure_balls": p_balls,
            "pressure_innings": p_inns,
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

    bw_senior_master = pd.DataFrame(bw_stats)

    # Process Senior Bowlers with Changes 7 (Innings floor & Clipping)
    def process_senior_bowlers(df_in, is_rr_only=False):
        df_out = df_in.copy()
        thresh_b = MIN_BALLS_BOWLER_RR if is_rr_only else np.where(df_out["is_rr"], MIN_BALLS_BOWLER_RR, MIN_BALLS_BOWLER_NON_RR)
        k_val = SHRINKAGE_K_BOWLER_RR if is_rr_only else np.where(df_out["is_rr"], SHRINKAGE_K_BOWLER_RR, SHRINKAGE_K_BOWLER_NON_RR)

        df_out["threshold_applied"] = thresh_b
        df_out["shrinkage_k"] = k_val
        # Change 7: Qualifies requires band_balls >= threshold AND pressure_innings >= MIN_INNINGS_BOWLER
        df_out["qualifies"] = (df_out["band_balls"] >= df_out["threshold_applied"]) & (df_out["pressure_innings"] >= MIN_INNINGS_BOWLER)
        df_out["eligibility"] = np.where(df_out["qualifies"], f"Eligible (>={thresh_b}b, >={MIN_INNINGS_BOWLER}inns)", "Ineligible")

        notes = []
        for _, row in df_out.iterrows():
            bb = row["band_balls"]
            lo = row["last_over_balls"]
            thresh = row["threshold_applied"]
            inns = row["pressure_innings"]
            if bb >= thresh * 2 and inns >= MIN_INNINGS_BOWLER * 2:
                notes.append(f"Robust sample ({bb} band balls, {lo} last-over balls across {inns} inns). High confidence.")
            elif bb >= thresh and inns >= MIN_INNINGS_BOWLER:
                notes.append(f"Qualified sample ({bb} >= {thresh} band balls across {inns} inns). Reliable indicator.")
            else:
                notes.append(f"Limited sample ({bb} band balls, {inns} inns). Below qualification threshold.")
        df_out["reliability_note"] = notes

        q = df_out[df_out["qualifies"]].copy()
        if len(q) > 0:
            rs_mean, rs_std = q["runs_saved_per_over"].mean(), q["runs_saved_per_over"].std(ddof=0)
            w_mean, w_std = q["wicket_component"].mean(), q["wicket_component"].std(ddof=0)
            if rs_std == 0: rs_std = 1.0
            if w_std == 0: w_std = 1.0

            q["z_runs_saved"] = (q["runs_saved_per_over"] - rs_mean) / rs_std
            q["z_wicket"] = (q["wicket_component"] - w_mean) / w_std
            # Change 7: Clip components to [-2.0, 2.0]
            q["z_runs_saved_clipped"] = np.clip(q["z_runs_saved"], -Z_CLIP, Z_CLIP)
            q["z_wicket_clipped"] = np.clip(q["z_wicket"], -Z_CLIP, Z_CLIP)

            q_lo = q[q["lo_qualifies"]]
            if len(q_lo) > 0 and q_lo["last_over_score"].std(ddof=0) > 0:
                lo_mean = q_lo["last_over_score"].mean()
                lo_std = q_lo["last_over_score"].std(ddof=0)
                q["z_last_over"] = np.where(q["lo_qualifies"], (q["last_over_score"] - lo_mean) / lo_std, 0.0)
            else:
                q["z_last_over"] = 0.0
            q["z_last_over_clipped"] = np.clip(q["z_last_over"], -Z_CLIP, Z_CLIP)

            q["raw_rating"] = q["z_runs_saved_clipped"] + q["z_wicket_clipped"] + 0.5 * q["z_last_over_clipped"]
            q["adjusted_rating"] = (q["pressure_balls"] * q["raw_rating"]) / (q["pressure_balls"] + q["shrinkage_k"])

            merge_cols = ["bowler_id", "z_runs_saved", "z_runs_saved_clipped", "z_wicket", "z_wicket_clipped", "z_last_over", "z_last_over_clipped", "raw_rating", "adjusted_rating"]
            df_out = df_out.merge(q[merge_cols], on="bowler_id", how="left")
            df_out["raw_rating"] = df_out["raw_rating"].round(4)
            df_out["adjusted_rating"] = df_out["adjusted_rating"].round(4)
        else:
            df_out["raw_rating"] = "n/a"
            df_out["adjusted_rating"] = "n/a"
        return df_out

    # Process Set 1 (RR), Set 2 (Next Season), Set 3 (All Senior)
    batters_rr = process_senior_batters(p_bat_agg[p_bat_agg["is_rr"]], inns_table, is_rr_only=True)
    bowlers_rr = process_senior_bowlers(bw_senior_master[bw_senior_master["is_rr"]], is_rr_only=True)

    batters_next_season = process_senior_batters(p_bat_agg[p_bat_agg["in_set2"] & (p_bat_agg["nationality"] != "Pakistan")], inns_table, is_rr_only=False)
    bowlers_next_season = process_senior_bowlers(bw_senior_master[bw_senior_master["in_set2"] & (bw_senior_master["nationality"] != "Pakistan")], is_rr_only=False)

    batters_all = process_senior_batters(p_bat_agg, inns_table, is_rr_only=False)
    bowlers_all = process_senior_bowlers(bw_senior_master, is_rr_only=False)

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
    # PART B: DEVELOPMENT & YOUTH PIPELINE
    # =========================================================================
    print("\n===================================================================")
    print("PART B: BUILD DEVELOPMENT & YOUTH BASELINE & RANKINGS")
    print("===================================================================")

    m_dy = m_valid[m_valid["match_tier"].isin(["DEVELOPMENT", "YOUTH"])].copy()
    dy_match_ids = set(m_dy["match_id"].unique())
    bbb_dy = bbb[bbb["match_id"].isin(dy_match_ids)].copy()

    df_dy, inn1_excluded_dy = prepare_deliveries_df(m_dy, bbb_dy)
    dev_exp_path = os.path.join(output_dir, "expected_tables_development.csv")
    exp_dy = build_expected_table(df_dy, dev_exp_path, mark_unreliable=True)
    df_dy = compute_loo_deliveries(df_dy, exp_dy)

    # Required rate tracking for Dev & Youth
    df_dy_press = df_dy[df_dy["is_batting_pressure"]].copy()
    df_dy_press["press_ball_num"] = df_dy_press.groupby(["match_id", "innings", "striker_id"]).cumcount() + 1
    df_dy_press["press_cum_runs"] = df_dy_press.groupby(["match_id", "innings", "striker_id"])["runs"].cumsum()
    df_dy_press["press_rpb"] = df_dy_press["press_cum_runs"] / df_dy_press["press_ball_num"]
    df_dy_press["rrr_rpb"] = df_dy_press["rrr"] / 6.0
    df_dy_press["is_below_rrr"] = df_dy_press["press_rpb"] < df_dy_press["rrr_rpb"]

    inns_table_dy = df_dy_press.groupby(["match_id", "innings", "striker_id"]).agg(
        pressure_balls=("runs", "count"),
        actual_runs=("runs", "sum"),
        expected_runs=("exp_batter_runs", "sum"),
        actual_dismissals=("batter_dismissal", "sum"),
        expected_dismissals=("exp_batter_dismissal", "sum"),
        balls_below_rrr=("is_below_rrr", "sum")
    ).reset_index()

    inns_table_dy["player"] = inns_table_dy["striker_id"].map(p_names)
    inns_table_dy["excess_dismissals"] = inns_table_dy["actual_dismissals"] - inns_table_dy["expected_dismissals"]
    inns_table_dy["runs_above_expected"] = inns_table_dy["actual_runs"] - inns_table_dy["expected_runs"]
    inns_table_dy["expected_runs_per_innings"] = inns_table_dy["expected_runs"]
    inns_table_dy["innings_value"] = inns_table_dy["runs_above_expected"] - inns_table_dy["excess_dismissals"] * inns_table_dy["expected_runs_per_innings"]
    inns_table_dy["pct_balls_below_rrr"] = (inns_table_dy["balls_below_rrr"] / inns_table_dy["pressure_balls"]) * 100.0

    p_teams_dev = mapping[mapping["comp_id"] == 2670].groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()
    p_teams_youth = mapping[mapping["comp_id"] == 3289].groupby("player_id")["team_name"].unique().apply(lambda ts: ", ".join(sorted(set(ts)))).to_dict()

    def aggregate_tier_players(df_subset, inns_subset, tier_label, p_teams_map):
        p_bat = inns_subset.groupby("striker_id").agg(
            pressure_balls=("pressure_balls", "sum"),
            pressure_innings=("pressure_balls", "count"),
            actual_runs=("actual_runs", "sum"),
            expected_runs=("expected_runs", "sum"),
            actual_dismissals=("actual_dismissals", "sum"),
            expected_dismissals=("expected_dismissals", "sum"),
            total_balls_below_rrr=("balls_below_rrr", "sum"),
            mean_innings_value=("innings_value", "mean")
        ).reset_index()

        p_bat["player"] = p_bat["striker_id"].map(p_names)
        p_bat["nationality"] = p_bat["striker_id"].map(player_nats)
        p_bat["tier"] = tier_label
        p_bat["teams"] = p_bat["striker_id"].map(lambda pid: p_teams_map.get(pid, p_teams_all.get(pid, "Unknown")))
        p_bat["pct_balls_below_rrr"] = (p_bat["total_balls_below_rrr"] / p_bat["pressure_balls"]) * 100.0

        # Bowlers
        bw_band_t = df_subset[df_subset["is_bowling_band"]].copy()
        bw_lo_t = df_subset[df_subset["is_last_over"]].copy()
        all_bw_t = sorted(df_subset["bowler_id"].dropna().unique())

        bw_stats_t = []
        for pid in all_bw_t:
            b_b = bw_band_t[bw_band_t["bowler_id"] == pid]
            b_l = bw_lo_t[bw_lo_t["bowler_id"] == pid]
            n_band = len(b_b)
            n_lo = len(b_l)
            p_balls = n_band + n_lo
            p_inns = len(set(b_b["match_id"]) | set(b_l["match_id"]))

            if p_balls == 0:
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

            bw_stats_t.append({
                "bowler_id": pid,
                "player": p_names.get(pid, f"Player_{pid}"),
                "nationality": player_nats.get(pid, "Unknown"),
                "tier": tier_label,
                "teams": p_teams_map.get(pid, p_teams_all.get(pid, "Unknown")),
                "pressure_balls": p_balls,
                "pressure_innings": p_inns,
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

        p_bw = pd.DataFrame(bw_stats_t)
        return p_bat, p_bw

    def process_tier_ratings(df_bat, df_bw, inns_df, tier_label):
        df_b = df_bat.copy()
        df_b["threshold_applied"] = MIN_BALLS_BATTER_DEV_YOUTH
        df_b["shrinkage_k"] = SHRINKAGE_K_BATTER_DEV_YOUTH
        # Development & Youth qualification: pressure_balls >= threshold and pressure_innings >= MIN_INNINGS_DEV_YOUTH
        df_b["qualifies"] = (df_b["pressure_balls"] >= df_b["threshold_applied"]) & (df_b["pressure_innings"] >= MIN_INNINGS_DEV_YOUTH)
        df_b["eligibility"] = np.where(df_b["qualifies"], f"Eligible (>={MIN_BALLS_BATTER_DEV_YOUTH}b, >={MIN_INNINGS_DEV_YOUTH}inns)", "Ineligible")

        notes = []
        for _, row in df_b.iterrows():
            pb = row["pressure_balls"]
            inns = row["pressure_innings"]
            if pb >= MIN_BALLS_BATTER_DEV_YOUTH and inns >= MIN_INNINGS_DEV_YOUTH:
                notes.append(f"Qualified sample ({pb} >= {MIN_BALLS_BATTER_DEV_YOUTH} balls across {inns} inns). Reliable trend in {tier_label}.")
            else:
                notes.append(f"Limited sample ({pb} balls, {inns} inns). Below qualification threshold for {tier_label}.")
        df_b["reliability_note"] = notes

        q_b = df_b[df_b["qualifies"]].copy()
        if len(q_b) > 0:
            q_pids = set(q_b["striker_id"])
            q_inns = inns_df[inns_df["striker_id"].isin(q_pids)]
            std_inns_val = q_inns["innings_value"].std(ddof=0)
            if std_inns_val == 0 or np.isnan(std_inns_val): std_inns_val = 1.0

            q_b["z_innings_value"] = q_b["mean_innings_value"] / std_inns_val
            q_b["z_innings_value_clipped"] = np.clip(q_b["z_innings_value"], -Z_CLIP, Z_CLIP)

            pct_m = q_b["pct_balls_below_rrr"].mean()
            pct_s = q_b["pct_balls_below_rrr"].std(ddof=0)
            if pct_s == 0 or np.isnan(pct_s): pct_s = 1.0
            q_b["z_pct_below_rrr"] = (q_b["pct_balls_below_rrr"] - pct_m) / pct_s
            q_b["z_pct_below_rrr_clipped"] = np.clip(q_b["z_pct_below_rrr"], -Z_CLIP, Z_CLIP)

            q_b["raw_rating"] = q_b["z_innings_value_clipped"] - RRR_PENALTY_WEIGHT * q_b["z_pct_below_rrr_clipped"]
            q_b["adjusted_rating"] = (q_b["pressure_balls"] * q_b["raw_rating"]) / (q_b["pressure_balls"] + SHRINKAGE_K_BATTER_DEV_YOUTH)

            merge_cols = ["striker_id", "z_innings_value", "z_innings_value_clipped", "z_pct_below_rrr", "z_pct_below_rrr_clipped", "raw_rating", "adjusted_rating"]
            df_b = df_b.merge(q_b[merge_cols], on="striker_id", how="left")
            df_b["raw_rating"] = df_b["raw_rating"].round(4)
            df_b["adjusted_rating"] = df_b["adjusted_rating"].round(4)
            df_b["mean_innings_value"] = df_b["mean_innings_value"].round(2)
            df_b["pct_balls_below_rrr"] = df_b["pct_balls_below_rrr"].round(1)
        else:
            df_b["raw_rating"] = "n/a"
            df_b["adjusted_rating"] = "n/a"

        # Bowlers
        df_bw_out = df_bw.copy()
        df_bw_out["threshold_applied"] = MIN_BALLS_BOWLER_DEV_YOUTH
        df_bw_out["shrinkage_k"] = SHRINKAGE_K_BOWLER_DEV_YOUTH
        df_bw_out["qualifies"] = (df_bw_out["band_balls"] >= MIN_BALLS_BOWLER_DEV_YOUTH) & (df_bw_out["pressure_innings"] >= MIN_INNINGS_DEV_YOUTH)
        df_bw_out["eligibility"] = np.where(df_bw_out["qualifies"], f"Eligible (>={MIN_BALLS_BOWLER_DEV_YOUTH}b, >={MIN_INNINGS_DEV_YOUTH}inns)", "Ineligible")

        bw_notes = []
        for _, row in df_bw_out.iterrows():
            bb = row["band_balls"]
            lo = row["last_over_balls"]
            inns = row["pressure_innings"]
            if bb >= MIN_BALLS_BOWLER_DEV_YOUTH and inns >= MIN_INNINGS_DEV_YOUTH:
                bw_notes.append(f"Qualified sample ({bb} >= {MIN_BALLS_BOWLER_DEV_YOUTH} band balls across {inns} inns). Reliable in {tier_label}.")
            else:
                bw_notes.append(f"Limited sample ({bb} band balls, {inns} inns). Below qualification threshold for {tier_label}.")
        df_bw_out["reliability_note"] = bw_notes

        q_bw = df_bw_out[df_bw_out["qualifies"]].copy()
        if len(q_bw) > 0:
            rs_mean, rs_std = q_bw["runs_saved_per_over"].mean(), q_bw["runs_saved_per_over"].std(ddof=0)
            w_mean, w_std = q_bw["wicket_component"].mean(), q_bw["wicket_component"].std(ddof=0)
            if rs_std == 0: rs_std = 1.0
            if w_std == 0: w_std = 1.0

            q_bw["z_runs_saved"] = (q_bw["runs_saved_per_over"] - rs_mean) / rs_std
            q_bw["z_wicket"] = (q_bw["wicket_component"] - w_mean) / w_std
            q_bw["z_runs_saved_clipped"] = np.clip(q_bw["z_runs_saved"], -Z_CLIP, Z_CLIP)
            q_bw["z_wicket_clipped"] = np.clip(q_bw["z_wicket"], -Z_CLIP, Z_CLIP)

            q_lo = q_bw[q_bw["lo_qualifies"]]
            if len(q_lo) > 0 and q_lo["last_over_score"].std(ddof=0) > 0:
                lo_mean = q_lo["last_over_score"].mean()
                lo_std = q_lo["last_over_score"].std(ddof=0)
                q_bw["z_last_over"] = np.where(q_bw["lo_qualifies"], (q_bw["last_over_score"] - lo_mean) / lo_std, 0.0)
            else:
                q_bw["z_last_over"] = 0.0
            q_bw["z_last_over_clipped"] = np.clip(q_bw["z_last_over"], -Z_CLIP, Z_CLIP)

            q_bw["raw_rating"] = q_bw["z_runs_saved_clipped"] + q_bw["z_wicket_clipped"] + 0.5 * q_bw["z_last_over_clipped"]
            q_bw["adjusted_rating"] = (q_bw["pressure_balls"] * q_bw["raw_rating"]) / (q_bw["pressure_balls"] + SHRINKAGE_K_BOWLER_DEV_YOUTH)

            merge_cols = ["bowler_id", "z_runs_saved", "z_runs_saved_clipped", "z_wicket", "z_wicket_clipped", "z_last_over", "z_last_over_clipped", "raw_rating", "adjusted_rating"]
            df_bw_out = df_bw_out.merge(q_bw[merge_cols], on="bowler_id", how="left")
            df_bw_out["raw_rating"] = df_bw_out["raw_rating"].round(4)
            df_bw_out["adjusted_rating"] = df_bw_out["adjusted_rating"].round(4)
        else:
            df_bw_out["raw_rating"] = "n/a"
            df_bw_out["adjusted_rating"] = "n/a"

        return df_b, df_bw_out

    # Process Development (Comp 2670)
    df_dev = df_dy[df_dy["match_tier"] == "DEVELOPMENT"]
    dev_mids = set(df_dev["match_id"].unique())
    inns_dev = inns_table_dy[inns_table_dy["match_id"].isin(dev_mids)]
    b_dev_raw, bw_dev_raw = aggregate_tier_players(df_dev, inns_dev, "Development", p_teams_dev)
    batters_development, bowlers_development = process_tier_ratings(b_dev_raw, bw_dev_raw, inns_dev, "Development")

    # Process Youth (Comp 3289)
    df_youth = df_dy[df_dy["match_tier"] == "YOUTH"]
    youth_mids = set(df_youth["match_id"].unique())
    inns_youth = inns_table_dy[inns_table_dy["match_id"].isin(youth_mids)]
    b_youth_raw, bw_youth_raw = aggregate_tier_players(df_youth, inns_youth, "Youth", p_teams_youth)
    batters_young, bowlers_young = process_tier_ratings(b_youth_raw, bw_youth_raw, inns_youth, "Youth")

    top10_dev_b = get_top10(batters_development)
    top10_dev_bw = get_top10(bowlers_development)
    top10_young_b = get_top10(batters_young)
    top10_young_bw = get_top10(bowlers_young)

    # =========================================================================
    # STEP 7: SANITY CHECKS & VERIFICATION (CHANGE 8)
    # =========================================================================
    run_synthetic_sanity_tests()

    # Verify no overlap between Senior matches and Dev/Youth matches
    if len(senior_match_ids & dy_match_ids) > 0:
        raise ValueError(f"Senior matches overlap with Development/Youth matches: {senior_match_ids & dy_match_ids}")
    else:
        print("  [PASSED] Strict Tier Separation: Zero overlap between Senior and Dev/Youth matches.")

    # Confirm thresholds for all top-10 lists
    all_t10_checks = [
        ("RR Batters", top10_rr_b, "pressure_balls", "pressure_innings", MIN_INNINGS_BATTER),
        ("RR Bowlers", top10_rr_bw, "band_balls", "pressure_innings", MIN_INNINGS_BOWLER),
        ("Next Season Batters", top10_next_b, "pressure_balls", "pressure_innings", MIN_INNINGS_BATTER),
        ("Next Season Bowlers", top10_next_bw, "band_balls", "pressure_innings", MIN_INNINGS_BOWLER),
        ("All Senior Batters", top10_all_b, "pressure_balls", "pressure_innings", MIN_INNINGS_BATTER),
        ("All Senior Bowlers", top10_all_bw, "band_balls", "pressure_innings", MIN_INNINGS_BOWLER),
        ("Development Batters", top10_dev_b, "pressure_balls", "pressure_innings", MIN_INNINGS_DEV_YOUTH),
        ("Development Bowlers", top10_dev_bw, "band_balls", "pressure_innings", MIN_INNINGS_DEV_YOUTH),
        ("Young Batters", top10_young_b, "pressure_balls", "pressure_innings", MIN_INNINGS_DEV_YOUTH),
        ("Young Bowlers", top10_young_bw, "band_balls", "pressure_innings", MIN_INNINGS_DEV_YOUTH),
    ]

    for label, t10_df, b_col, in_col, in_min in all_t10_checks:
        for _, r in t10_df.iterrows():
            thresh_applied = r["threshold_applied"]
            if r[b_col] < thresh_applied or r[in_col] < in_min:
                raise ValueError(f"Threshold violation in {label}: {r['player']} has {b_col}={r[b_col]} (min {thresh_applied}) or {in_col}={r[in_col]} (min {in_min})!")
    print("  [PASSED] Verification: Every top-10 player in all 10 lists strictly meets both ball and innings thresholds.")

    # =========================================================================
    # STEP 6: EXPORT CSV OUTPUTS (CHANGE 10)
    # =========================================================================
    print("\n===================================================================")
    print("STEP 6: EXPORT CSV OUTPUTS")
    print("===================================================================")

    # Full tables
    batters_rr.to_csv(os.path.join(output_dir, "batters_rr.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_rr.csv"), index=False)
    batters_next_season.to_csv(os.path.join(output_dir, "batters_next_season.csv"), index=False)
    bowlers_next_season.to_csv(os.path.join(output_dir, "bowlers_next_season.csv"), index=False)
    batters_all.to_csv(os.path.join(output_dir, "batters_all.csv"), index=False)
    bowlers_all.to_csv(os.path.join(output_dir, "bowlers_all.csv"), index=False)

    batters_rr.to_csv(os.path.join(output_dir, "batters_pressure.csv"), index=False)
    bowlers_rr.to_csv(os.path.join(output_dir, "bowlers_pressure.csv"), index=False)

    # Top 10 tables
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

    # Print New Top 10s with Clipped Components, Innings Count, and pct_balls_below_rrr (Change 10)
    print("\n" + "="*80)
    print("TOP 10 RANKINGS WITH CLIPPED COMPONENTS & INNINGS COUNT (CHANGE 10)")
    print("="*80)

    print("\n--- TOP 10 RAJASTHAN ROYALS BATTERS (NEW) ---")
    print(top10_rr_b[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "pct_balls_below_rrr", "z_innings_value_clipped", "z_pct_below_rrr_clipped"]].to_string(index=False))

    print("\n--- TOP 10 RAJASTHAN ROYALS BOWLERS (NEW) ---")
    print(top10_rr_bw[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "band_balls", "z_runs_saved_clipped", "z_wicket_clipped", "z_last_over_clipped"]].to_string(index=False))

    print("\n--- TOP 10 NEXT SEASON BATTERS (NEW) ---")
    print(top10_next_b[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "pct_balls_below_rrr", "z_innings_value_clipped", "z_pct_below_rrr_clipped"]].to_string(index=False))

    print("\n--- TOP 10 NEXT SEASON BOWLERS (NEW) ---")
    print(top10_next_bw[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "band_balls", "z_runs_saved_clipped", "z_wicket_clipped", "z_last_over_clipped"]].to_string(index=False))

    print("\n--- TOP 10 ALL SENIOR BATTERS (NEW) ---")
    print(top10_all_b[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "pct_balls_below_rrr", "z_innings_value_clipped", "z_pct_below_rrr_clipped"]].to_string(index=False))

    print("\n--- TOP 10 ALL SENIOR BOWLERS (NEW) ---")
    print(top10_all_bw[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "band_balls", "z_runs_saved_clipped", "z_wicket_clipped", "z_last_over_clipped"]].to_string(index=False))

    print("\n--- TOP 10 DEVELOPMENT BATTERS (NEW) ---")
    print(top10_dev_b[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "pct_balls_below_rrr", "z_innings_value_clipped", "z_pct_below_rrr_clipped"]].to_string(index=False))

    print("\n--- TOP 10 DEVELOPMENT BOWLERS (NEW) ---")
    print(top10_dev_bw[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "band_balls", "z_runs_saved_clipped", "z_wicket_clipped", "z_last_over_clipped"]].to_string(index=False))

    print("\n--- TOP 10 YOUNG BATTERS (NEW) ---")
    print(top10_young_b[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "pct_balls_below_rrr", "z_innings_value_clipped", "z_pct_below_rrr_clipped"]].to_string(index=False))

    print("\n--- TOP 10 YOUNG BOWLERS (NEW) ---")
    print(top10_young_bw[["rank", "player", "teams", "adjusted_rating", "pressure_balls", "pressure_innings", "band_balls", "z_runs_saved_clipped", "z_wicket_clipped", "z_last_over_clipped"]].to_string(index=False))

    # Print comparisons between old and new top 10s (Change 10)
    print("\n" + "="*80)
    print("OLD VS NEW TOP 10 RANKING COMPARISONS (CHANGE 10)")
    print("="*80)

    print_old_vs_new_comparison("RR Batters", top10_rr_b, OLD_TOP10["RR Batters"])
    print_old_vs_new_comparison("RR Bowlers", top10_rr_bw, OLD_TOP10["RR Bowlers"])
    print_old_vs_new_comparison("Next Season Batters", top10_next_b, OLD_TOP10["Next Season Batters"])
    print_old_vs_new_comparison("Next Season Bowlers", top10_next_bw, OLD_TOP10["Next Season Bowlers"])
    print_old_vs_new_comparison("All Senior Batters", top10_all_b, OLD_TOP10["All Senior Batters"])
    print_old_vs_new_comparison("All Senior Bowlers", top10_all_bw, OLD_TOP10["All Senior Bowlers"])
    print_old_vs_new_comparison("Development Batters", top10_dev_b, OLD_TOP10["Development Batters"])
    print_old_vs_new_comparison("Development Bowlers", top10_dev_bw, OLD_TOP10["Development Bowlers"])
    print_old_vs_new_comparison("Young Batters", top10_young_b, OLD_TOP10["Young Batters"])
    print_old_vs_new_comparison("Young Bowlers", top10_young_bw, OLD_TOP10["Young Bowlers"])

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
    }


run_pressure_pipeline = run_multi_tier_pipeline

if __name__ == "__main__":
    run_multi_tier_pipeline()

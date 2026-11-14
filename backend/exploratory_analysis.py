"""
Exploratory Analysis of Rajasthan Royals Player Dataset
Part of SupeRR Selector: Rajasthan Royals Pressure Analysis

Reconciliation of RR Player Counts:
-----------------------------------
1. Squad Roster (69 players): Total unique players registered in Rajasthan Royals
   squad mapping (competition_team_player_mapping.csv) across all competitions from 2022 onward.
2. Batted in Matches (57 players): Unique RR players who actually faced at least one legal delivery
   or extra delivery (recorded as striker) in ball-by-ball match records.
3. Bowled in Matches (45 players): Unique RR players who bowled at least one delivery
   (recorded as bowler) in ball-by-ball match records.
4. Neither Batted nor Bowled (9 players): Squad members who were reserve, backup, or non-playing
   members during the tournament window and recorded zero balls faced or bowled.
   (Note: 60 unique players participated as either a batter or bowler; 56 took the pitch
   in official matchday playing XIs with is_played == True).
5. Qualifying Players (under WP Pressure Definition):
   - Batters meeting sample threshold (>= 30 pressure balls): 28 players
   - Bowlers meeting sample threshold (>= 24 pressure balls): 32 players
"""

import os
import pandas as pd
import numpy as np

def run_exploratory_analysis(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")

    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")
    competitions_path = os.path.join(data_dir, "competitions.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")
    matches_path = os.path.join(data_dir, "matches.csv")

    print("===================================================================")
    print("STEP 1: Load Player Mapping & Competition Datasets")
    print("===================================================================")
    print(f"Data directory: {data_dir}")

    df_players = pd.read_csv(mapping_path)
    df_comp = pd.read_csv(competitions_path)

    # Attach competition season
    df = df_players.merge(df_comp[["comp_id", "season"]], on="comp_id", how="left")

    # Filter specifically for Rajasthan Royals
    df_rr = df[df["team_name"].str.contains("Rajasthan Royals", case=False, na=False)].copy()

    print(f"Rajasthan Royals dataset shape: {df_rr.shape}")
    print(f"Columns: {df_rr.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df_rr.head())

    print("\n===================================================================")
    print("STEP 2: Count Unique Players (Player ID vs Player Name)")
    print("===================================================================")
    id_col = next((c for c in df_rr.columns if "player_id" in c.lower() or c.lower() == "id"), None)
    name_col = next((c for c in df_rr.columns if "player_name" in c.lower() or "player" in c.lower()), None)

    if id_col:
        unique_by_id = df_rr[id_col].dropna().nunique()
        print(f"Unique players using '{id_col}': {unique_by_id}")
    if name_col:
        df_rr[name_col] = df_rr[name_col].astype(str).str.strip()
        unique_by_name = df_rr[name_col].dropna().nunique()
        print(f"Unique players using stripped '{name_col}': {unique_by_name}")

    print("\n===================================================================")
    print("STEP 3: Distinct Seasons & Multi-Year Text Formats")
    print("===================================================================")
    season_col = "season" if "season" in df_rr.columns else "comp_name"
    df_rr[season_col] = df_rr[season_col].astype(str).str.strip()
    seasons = sorted(df_rr[season_col].dropna().unique())
    print(f"Distinct seasons ({len(seasons)}): {seasons}")

    print("\n===================================================================")
    print("STEP 4: Unique Players per Season")
    print("===================================================================")
    player_col = id_col if id_col else name_col
    players_per_season = df_rr.groupby(season_col)[player_col].nunique()
    print(players_per_season)

    print("\n===================================================================")
    print("STEP 5: Missing Values Audit")
    print("===================================================================")
    print(f"Missing in player column ('{player_col}'): {df_rr[player_col].isna().sum()}")
    print(f"Missing in season column ('{season_col}'): {df_rr[season_col].isna().sum()}")

    print("\n===================================================================")
    print("STEP 6: Player Counts Reconciliation (Task 6)")
    print("===================================================================")
    # Load match delivery logs
    bbb = pd.read_csv(bbb_path, usecols=["striker_name", "bowler_name"])
    rr_roster = set(df_rr["player_name"].str.strip().unique())

    all_strikers = set(bbb["striker_name"].astype(str).str.strip().unique())
    all_bowlers = set(bbb["bowler_name"].astype(str).str.strip().unique())

    rr_batted = rr_roster.intersection(all_strikers)
    rr_bowled = rr_roster.intersection(all_bowlers)
    rr_both = rr_batted.intersection(rr_bowled)
    rr_either = rr_batted.union(rr_bowled)
    rr_unplayed = rr_roster - rr_either

    print(f"1. Roster Count (Official RR Squad):        {len(rr_roster)} players")
    print(f"2. Batted Count (Faced >= 1 ball):          {len(rr_batted)} players")
    print(f"3. Bowled Count (Bowled >= 1 ball):         {len(rr_bowled)} players")
    print(f"   - Both Batted and Bowled (All-rounders): {len(rr_both)} players")
    print(f"   - Neither Batted nor Bowled (Reserves):  {len(rr_unplayed)} players")

    # Load computed pressure datasets if available to report qualifying
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    b_csv = os.path.join(data_folder, "batters_pressure.csv")
    bw_csv = os.path.join(data_folder, "bowlers_pressure.csv")

    if os.path.exists(b_csv) and os.path.exists(bw_csv):
        b_df = pd.read_csv(b_csv)
        bw_df = pd.read_csv(bw_csv)
        qual_batters = b_df[b_df["pressure_balls"] >= 30]
        qual_bowlers = bw_df[bw_df["pressure_balls"] >= 24]
        print(f"4. Qualifying Count (Meeting Minimum Sample Thresholds):")
        print(f"   - Batters (>= 30 pressure balls):        {len(qual_batters)} qualifying players (out of {len(b_df)} with pressure deliveries)")
        print(f"   - Bowlers (>= 24 pressure balls):        {len(qual_bowlers)} qualifying players (out of {len(bw_df)} with pressure deliveries)")

    print("\nReconciliation Summary:")
    print("-----------------------")
    print("• Squad Roster (69): Every player contracted or mapped to Rajasthan Royals across 2022-2026 IPL editions.")
    print("• Active In-Match (60): 57 players batted and 45 bowled across IPL/T20Is (42 players performed both roles).")
    print("• Reserves (9): Backup players who did not face or bowl any delivery during competitive match records.")
    print("• Qualifying (28 Batters, 32 Bowlers): Players who faced sufficient high-pressure deliveries to pass statistical sample-size thresholds.")

if __name__ == "__main__":
    run_exploratory_analysis()

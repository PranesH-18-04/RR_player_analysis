import os
import pandas as pd

def analyze_rajasthan_royals_players(data_dir=None):
    """
    Step 1: Load player mapping and competitions dataset.
    Step 2: Filter specifically for Rajasthan Royals.
    Step 3: Count unique players using player_id if available, otherwise player_name (whitespace stripped).
    Step 4: List distinct seasons/years, handling text representations like '2018/19' or '2022/23'.
    Step 5: Show unique players per season.
    Step 6: Report missing values in player and season columns.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")

    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")
    competitions_path = os.path.join(data_dir, "competitions.csv")

    print(f"Loading data from: {data_dir}")
    df_players = pd.read_csv(mapping_path)
    df_comp = pd.read_csv(competitions_path)

    # Merge on comp_id to attach season
    df = df_players.merge(df_comp[["comp_id", "season"]], on="comp_id", how="left")

    # Filter for Rajasthan Royals
    if "team_name" in df.columns:
        df = df[df["team_name"].str.contains("Rajasthan Royals", case=False, na=False)].copy()

    print("\n--- 1. Dataset Shape, Columns, and First 5 Rows ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Detect player ID or name column
    id_col = next((c for c in df.columns if "player_id" in c.lower() or c.lower() == "id"), None)
    name_col = next((c for c in df.columns if "player_name" in c.lower() or "player" in c.lower()), None)

    print("\n--- 2. Unique Players Count ---")
    if id_col:
        player_col = id_col
        unique_players = df[player_col].dropna().nunique()
        print(f"Using Player ID column: '{player_col}'")
    elif name_col:
        player_col = name_col
        df[player_col] = df[player_col].astype(str).str.strip()
        unique_players = df[player_col].dropna().nunique()
        print(f"Using stripped Player Name column: '{player_col}'")
    else:
        raise KeyError("No valid player ID or name column found.")
    print(f"Total Unique Rajasthan Royals Players: {unique_players}")

    print("\n--- 3. Distinct Years / Seasons ---")
    season_col = next((c for c in df.columns if "season" in c.lower() or "year" in c.lower()), "season")
    df[season_col] = df[season_col].astype(str).str.strip()
    distinct_seasons = sorted(df[season_col].dropna().unique())
    print(f"Distinct Seasons ({len(distinct_seasons)}): {distinct_seasons}")

    print("\n--- 4. Unique Players Appeared in Each Season ---")
    players_per_season = df.groupby(season_col)[player_col].nunique()
    print(players_per_season)

    print("\n--- 5. Missing Values Check ---")
    missing_player = df[player_col].isna().sum()
    missing_season = df[season_col].isna().sum()
    print(f"Missing values in '{player_col}': {missing_player}")
    print(f"Missing values in '{season_col}': {missing_season}")

if __name__ == "__main__":
    analyze_rajasthan_royals_players()

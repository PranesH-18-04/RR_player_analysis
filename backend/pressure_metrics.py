import os
import json
import pandas as pd
import numpy as np

def generate_pressure_metrics(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "Downloads", "6af458274c1e8_data_pack", "Data Pack")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    os.makedirs(output_dir, exist_ok=True)

    mapping_path = os.path.join(data_dir, "competition_team_player_mapping.csv")
    matches_path = os.path.join(data_dir, "matches.csv")
    bbb_path = os.path.join(data_dir, "ball_by_ball.csv")

    mapping = pd.read_csv(mapping_path)
    rr_players = set(mapping[mapping["team_name"] == "Rajasthan Royals"]["player_name"].str.strip().unique())

    matches = pd.read_csv(matches_path)
    bbb = pd.read_csv(bbb_path)

    # Filter matches
    matches["year"] = pd.to_datetime(matches["match_date"]).dt.year
    valid_matches = matches[
        (matches["year"] >= 2022)
        & matches["win_team_id"].notna()
        & (matches["win_team_id"] != 0)
        & ~matches["result"].str.contains(r"no\s*result|tie|abandon", case=False, na=False)
    ].copy()

    bbb["total_ball_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    inn1_totals = bbb[bbb["innings"] == 1].groupby("match_id")["total_ball_runs"].sum().rename("inn1_total")
    valid_matches = valid_matches.merge(inn1_totals, on="match_id", how="inner")

    valid_matches["target"] = np.where(
        valid_matches["revised_target"].notna() & (valid_matches["revised_target"] > 0),
        valid_matches["revised_target"],
        valid_matches["inn1_total"] + 1,
    )
    valid_matches["total_balls"] = np.where(
        valid_matches["second_inn_revised_overs"].notna() & (valid_matches["second_inn_revised_overs"] > 0),
        valid_matches["second_inn_revised_overs"],
        np.where(
            valid_matches["first_inn_revised_overs"].notna() & (valid_matches["first_inn_revised_overs"] > 0),
            valid_matches["first_inn_revised_overs"],
            120,
        ),
    )

    df = bbb.merge(valid_matches[["match_id", "target", "total_balls", "win_team_id"]], on="match_id")
    df["cum_runs"] = df.groupby(["match_id", "innings"])["total_ball_runs"].cumsum()
    df["runs_needed"] = df["target"] - df["cum_runs"]
    df["balls_remaining"] = df["total_balls"] - df["innings_ball_num"]
    df["rrr"] = np.where(df["balls_remaining"] > 0, df["runs_needed"] * 6.0 / df["balls_remaining"], np.nan)
    df["is_pressure"] = (df["overid"] >= 16) | ((df["innings"] == 2) & (df["rrr"] >= 9.0))

    # 1. Batters Metrics
    df["striker_name_clean"] = df["striker_name"].astype(str).str.strip()
    rr_batters = df[df["striker_name_clean"].isin(rr_players)].copy()

    batter_rows = []
    for p in sorted(rr_batters["striker_name_clean"].unique()):
        pdf = rr_batters[rr_batters["striker_name_clean"] == p]
        norm = pdf[~pdf["is_pressure"]]
        press = pdf[pdf["is_pressure"]]

        n_balls, p_balls = len(norm), len(press)
        if n_balls == 0 and p_balls == 0:
            continue
        p_inns = press["match_id"].nunique()

        norm_sr = round((norm["runs"].sum() / n_balls) * 100, 1) if n_balls > 0 else np.nan
        press_sr = round((press["runs"].sum() / p_balls) * 100, 1) if p_balls > 0 else np.nan
        sr_diff = round(press_sr - norm_sr, 1) if (pd.notna(press_sr) and pd.notna(norm_sr)) else np.nan

        norm_bndry = round(((norm["is_four"] | norm["is_six"]).sum() / n_balls) * 100, 1) if n_balls > 0 else np.nan
        press_bndry = round(((press["is_four"] | press["is_six"]).sum() / p_balls) * 100, 1) if p_balls > 0 else np.nan

        if p_balls >= 60:
            rel = f"Robust sample ({p_balls} balls across {p_inns} inns). Highly statistically reliable."
        elif p_balls >= 30:
            rel = f"Moderate sample ({p_balls} balls across {p_inns} inns). Reliable trend indicators."
        else:
            rel = f"Small sample warning ({p_balls} balls). High variance; interpret with caution."

        batter_rows.append({
            "player_name": p,
            "pressure_balls": p_balls,
            "pressure_innings": p_inns,
            "normal_strike_rate": norm_sr if pd.notna(norm_sr) else "n/a",
            "pressure_strike_rate": press_sr if pd.notna(press_sr) else "n/a",
            "sr_difference": sr_diff if pd.notna(sr_diff) else "n/a",
            "normal_boundary_pct": norm_bndry if pd.notna(norm_bndry) else "n/a",
            "pressure_boundary_pct": press_bndry if pd.notna(press_bndry) else "n/a",
            "reliability_note": rel,
        })

    # 2. Bowlers Metrics
    df["bowler_extras"] = np.where(df["extras_type"].isin(["byes", "legbyes", "Penalty"]), 0, df["extras1_runs"])
    df["bowler_runs_conceded"] = df["runs"] + df["bowler_extras"]
    df["is_legal_ball"] = ~df["extras_type"].isin(["wide", "Overstep"])
    df["bowler_wicket"] = df["is_wicket"] & ~df["dismissal_type"].astype(str).str.contains("run out", case=False, na=False)

    df["bowler_name_clean"] = df["bowler_name"].astype(str).str.strip()
    rr_bowlers = df[df["bowler_name_clean"].isin(rr_players)].copy()

    bowler_rows = []
    for p in sorted(rr_bowlers["bowler_name_clean"].unique()):
        pdf = rr_bowlers[rr_bowlers["bowler_name_clean"] == p]
        norm = pdf[~pdf["is_pressure"]]
        press = pdf[pdf["is_pressure"]]

        n_balls, p_balls = len(norm), len(press)
        if n_balls == 0 and p_balls == 0:
            continue

        p_legal = press["is_legal_ball"].sum()
        n_legal = norm["is_legal_ball"].sum()

        p_overs = round(p_legal / 6.0, 1) if p_legal > 0 else 0.0

        norm_econ = round((norm["bowler_runs_conceded"].sum() / (n_legal / 6.0)), 2) if n_legal >= 6 else np.nan
        press_econ = round((press["bowler_runs_conceded"].sum() / (p_legal / 6.0)), 2) if p_legal >= 6 else np.nan
        econ_diff = round(press_econ - norm_econ, 2) if (pd.notna(press_econ) and pd.notna(norm_econ)) else np.nan

        norm_dots = round(((norm["runs"] == 0) & (norm["extras1_runs"] == 0)).sum() / n_balls * 100, 1) if n_balls > 0 else np.nan
        press_dots = round(((press["runs"] == 0) & (press["extras1_runs"] == 0)).sum() / p_balls * 100, 1) if p_balls > 0 else np.nan

        wkts_press = int(press["bowler_wicket"].sum())

        if p_balls >= 60:
            rel = f"Robust sample ({p_balls} balls, {p_overs} ov). High confidence in pressure economy."
        elif p_balls >= 24:
            rel = f"Moderate sample ({p_balls} balls, {p_overs} ov). Useful contextual indicator."
        else:
            rel = f"Small sample warning ({p_balls} balls). High variance; interpret with caution."

        bowler_rows.append({
            "player_name": p,
            "pressure_balls": p_balls,
            "pressure_overs": p_overs,
            "normal_economy": norm_econ if pd.notna(norm_econ) else "n/a",
            "pressure_economy": press_econ if pd.notna(press_econ) else "n/a",
            "economy_difference": econ_diff if pd.notna(econ_diff) else "n/a",
            "normal_dot_pct": norm_dots if pd.notna(norm_dots) else "n/a",
            "pressure_dot_pct": press_dots if pd.notna(press_dots) else "n/a",
            "wickets_pressure": wkts_press,
            "reliability_note": rel,
        })

    # Save CSVs
    b_df = pd.DataFrame(batter_rows)
    bw_df = pd.DataFrame(bowler_rows)

    b_path = os.path.join(output_dir, "batters_pressure.csv")
    bw_path = os.path.join(output_dir, "bowlers_pressure.csv")
    b_df.to_csv(b_path, index=False)
    bw_df.to_csv(bw_path, index=False)

    print(f"Saved {len(b_df)} batters to {b_path}")
    print(f"Saved {len(bw_df)} bowlers to {bw_path}")

if __name__ == "__main__":
    generate_pressure_metrics()

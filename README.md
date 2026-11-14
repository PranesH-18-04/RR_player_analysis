# SupeRR Selector: Rajasthan Royals Pressure Analysis & Win Probability Model

A data science system and interactive analytics dashboard built for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This repository evaluates historical T20 performance (IPL and T20 International data, 2022 onward) to measure how batters and bowlers perform in high-pressure match situations compared with their usual baselines.

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Single-page interactive RR Pressure Analysis Dashboard
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster exploration, unique player counts & season parsing
│   ├── win_probability_model.py   # Second-innings win probability model (Logistic Regression)
│   └── pressure_metrics.py        # Pressure vs normal metric extraction for RR players
│
├── data/
│   ├── batters_pressure.csv       # Computed pressure metrics for 57 RR batters
│   ├── bowlers_pressure.csv       # Computed pressure metrics for 45 RR bowlers
│   └── win_prob_curves.png        # In-play win probability validation curves
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## 1. Exploratory Player Analysis

* **Scope:** Rajasthan Royals players across IPL and international series from 2022 onward.
* **Squad Roster Size:** **69 unique players** across **5 seasons** (`2022`, `2023`, `2024`, `2025`, `2026`).
* **Active Matchday Appearances:** **56 unique players** took the pitch (`is_played == True`).
* **Missing Values:** Zero missing values in player identification and season columns.
* **Run Script:**
  ```bash
  python backend/exploratory_analysis.py
  ```

---

## 2. Second-Innings Win Probability Model (Chasing Side)

* **Objective:** Predict in-play win probability after every ball in the second innings.
* **Features:**
  1. `runs_needed`: Target minus cumulative runs scored. Handles D/L-affected targets via `revised_target` when present.
  2. `balls_remaining`: Total balls allowed (`second_inn_revised_overs` when revised, else 120) minus delivery sequence.
  3. `wickets_in_hand`: $10 - \text{cumulative wickets}$.
  4. `rrr`: Required run rate $(\text{runs\_needed} \times 6 / \text{balls\_remaining})$.
* **Model:** Logistic Regression with a time-based and match-grouped split:
  * **Train Set (2022–2024):** 501 matches (55,602 ball events)
  * **Test Set (2025 onward):** 337 matches (36,556 ball events)
* **Performance on Unseen Test Set:**
  * **Log Loss:** `0.4169` (well below uninformative benchmark $0.6931$)
  * **Brier Score:** `0.1354`
  * **Monotonic Calibration:** Validated across 10 probability bins from $0.0$ to $1.0$.
* **Run Script:**
  ```bash
  python backend/win_probability_model.py
  ```

---

## 3. Pressure Analytics & Metrics

* **Pressure Definition:**
  * Innings 2: Deliveries in death overs (overs 16–20) **OR** where required run rate $\ge 9.00$ runs/over.
  * Innings 1: Deliveries in death overs (overs 16–20).
* **Batter Metrics:** `pressure_balls`, `pressure_innings`, `normal_strike_rate`, `pressure_strike_rate`, `sr_difference`, `normal_boundary_pct`, `pressure_boundary_pct`, `reliability_note`.
* **Bowler Metrics:** `pressure_balls`, `pressure_overs`, `normal_economy`, `pressure_economy`, `economy_difference`, `normal_dot_pct`, `pressure_dot_pct`, `wickets_pressure`, `reliability_note`.
* **Run Script:**
  ```bash
  python backend/pressure_metrics.py
  ```

---

## 4. Interactive HTML Dashboard

A responsive single-page web app built with vanilla JavaScript, CSS, and Chart.js matching the Rajasthan Royals visual identity:
* **Color Palette:** Primary Pink (`#E91E8C`), Deep Blue (`#1A3C8F`), Light Blue (`#EAF2FB`), and Dark Text (`#14213D`).
* **Player Selector:** Dropdown grouped into RR Batters and RR Bowlers with instant profile synchronization.
* **Sample-Size Warning Badges:** Automatically flags players with fewer than 30 pressure balls (batters) or 24 pressure balls (bowlers).
* **Difference Indicator:** Large differential number colored pink for favorable outcomes (strike rate increases or economy drops) and blue for unfavorable trends, paired with an auto-generated plain-English summary.
* **Comparison Bar Chart:** Chart.js grouped bar chart comparing normal vs. pressure metrics.
* **Sortable Roster Table:** Toggle between batters and bowlers with column sorting, search filter, and alternating `#EAF2FB` row shading.
* **How to View:**
  Double click [index.html](index.html) or [frontend/index.html](frontend/index.html) to open in any web browser.

---

## License & Usage

Data pack for competition use only. Not for publication.
© Rajasthan Royals • SupeRR Selector Season 2.

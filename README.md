# SupeRR Selector: Rajasthan Royals Pressure Analysis & Multi-Set Ranking Engine

A decision science and talent scouting system engineered for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure using game-state expected-value baselines constructed from all qualifying IPL and T20 International matches (2022 onward).

The repository features three independent, internally consistent ranking sets with tiered eligibility thresholds:
* **Set 1 (RR Players):** Franchise-specific ranking of players who appeared for Rajasthan Royals (`team_id == 3`).
* **Set 2 (Next Season Eligible):** Strategic proxy ranking of players who appeared in the most recent two seasons (`2025` and `2026`), including active RR players. *(Note: This serves as a realistic talent availability proxy, not official squad registration).*
* **Set 3 (All Players):** Global ranking across all players in the dataset regardless of team or nationality (with Pakistan players fully maintained in the analytical pool).

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Standalone interactive dashboard (3-way pool toggle, Pakistan filter, RR palette)
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory analysis
│   ├── win_probability_model.py   # Second-innings logistic regression model & swing thresholding
│   └── pressure_metrics.py        # Tiered threshold engine, 3 ranking sets, Bayesian shrinkage & exports
│
├── data/
│   ├── expected_tables.csv        # 413 empirical game-state expected value buckets
│   ├── top10_rr_batters.csv       # Top 10 Batters (Set 1: RR Players)
│   ├── top10_rr_bowlers.csv       # Top 10 Bowlers (Set 1: RR Players)
│   ├── top10_next_season_batters.csv # Top 10 Batters (Set 2: Next Season Eligible)
│   ├── top10_next_season_bowlers.csv # Top 10 Bowlers (Set 2: Next Season Eligible)
│   ├── top10_all_batters.csv      # Top 10 Batters (Set 3: All Players - Global)
│   ├── top10_all_bowlers.csv      # Top 10 Bowlers (Set 3: All Players - Global)
│   ├── batters_rr.csv             # Full per-player metrics for RR batters
│   ├── bowlers_rr.csv             # Full per-player metrics for RR bowlers
│   ├── batters_next_season.csv    # Full per-player metrics for Next Season batters
│   ├── bowlers_next_season.csv    # Full per-player metrics for Next Season bowlers
│   ├── batters_all.csv            # Full per-player metrics for all global batters
│   ├── bowlers_all.csv            # Full per-player metrics for all global bowlers
│   ├── batters_pressure.csv       # Backward-compatible RR batters table
│   ├── bowlers_pressure.csv       # Backward-compatible RR bowlers table
│   └── win_prob_curves.png        # In-play win probability validation curves
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## 1. Data Audit & Confirmation (Step 1)

* **Team Confirmation:** Rajasthan Royals is confirmed from `teams.csv` and `matches.csv` as **`team_id == 3`** (appearing as home/away team `"Rajasthan Royals"` in all competitive records).
* **Season Confirmation:** The data spans 8 distinct seasons (`2022`, `2022/23`, `2023`, `2023/24`, `2024`, `2024/25`, `2025`, `2026`). The most recent season is **`2026`**, preceded by **`2025`**.
* **Nationality Mapping:** Identified for all 1,217 players based on official international representation in `competition_team_player_mapping.csv`, with domestic Indian talent mapped to `India` and verified overseas players accurately categorized.
* **Pakistan Players Preserved:** Pakistan players are **fully preserved** in the analytical pool (68 players). The dashboard includes a dedicated filter allowing scouts to view rankings with or without Pakistan players without altering the statistical baseline.

### Player Breakdown by Nationality
| Nationality | Count | Nationality | Count | Nationality | Count |
| :--- | :---: | :--- | :---: | :--- | :---: |
| **India** | 301 | **West Indies** | 53 | **Netherlands** | 35 |
| **Pakistan** | 68 | **South Africa** | 53 | **Ireland** | 34 |
| **United Arab Emirates** | 68 | **Australia** | 50 | **Nepal** | 24 |
| **Afghanistan** | 67 | **New Zealand** | 47 | **Singapore** | 23 |
| **Sri Lanka** | 63 | **Zimbabwe** | 47 | **Canada** | 19 |
| **England** | 62 | **Oman** | 39 | **Scotland / USA** | 18 each |
| **Bangladesh** | 58 | **Hong Kong** | 37 | **Kuwait / Japan / Namibia** | 17 / 15 / 1 |

---

## 2. Three Ranking Sets & Tiered Thresholds (Steps 2 & 3)

### Definitions of the Sets
1. **Set 1 (RR Players):** Every player who has appeared for Rajasthan Royals (`team_id == 3`) in mapping or ball-by-ball match play.
2. **Set 2 (Next Season Eligible):** Every player who appeared in the most recent two seasons in the data (`2025` and `2026`), including Rajasthan Royals players. *This serves as an objective proxy for talent currently active and eligible for upcoming tournament cycles, rather than an official roster registration.*
3. **Set 3 (All Players):** Every player in the dataset regardless of team, franchise, or nationality.

### Tiered Minimum Sample Thresholds
To balance robust statistical confidence for general scouting with nuanced franchise evaluation, tiered delivery thresholds are enforced:
* **RR Players (Set 1, and RR players in Sets 2 & 3):**
  * Batters: $\ge \mathbf{30}$ pressure deliveries faced.
  * Bowlers: $\ge \mathbf{24}$ deliveries in the RRR 8–12 chase band.
  * Bayesian Shrinkage: $k = 30$ (batters), $k = 24$ (bowlers).
* **Non-RR Players (Outside RR in Sets 2 & 3):**
  * Batters: raised to $\ge \mathbf{60}$ pressure deliveries faced.
  * Bowlers: raised to $\ge \mathbf{48}$ deliveries in the RRR 8–12 chase band.
  * Bayesian Shrinkage: $k = 60$ (batters), $k = 48$ (bowlers).

### Threshold Removal Audit
| Ranking Set | Total Pool Size | Role | Evaluated Players | Qualified Players | Removed by Threshold |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Set 1: RR Players** | **69** | Batters | 57 | **33** | 24 |
| | | Bowlers | 45 | **34** | 11 |
| **Set 2: Next Season Eligible** | **777** | Batters | 636 | **281** | 355 |
| | | Bowlers | 478 | **191** | 287 |
| **Set 3: All Players (Global)** | **1,217** | Batters | 888 | **317** | 571 |
| | | Bowlers | 646 | **213** | 433 |

---

## 3. Expected-Value Baselines & Pressure Definitions (Steps 4 & 5)

A single unified expected-value baseline was constructed across all **198,032 deliveries** in 838 valid matches, ensuring all three ranking sets share identical expectations:

* **Second Innings (Chase):** Partitioned by `balls_remaining` (1–6, 7–12, 13–24, 25–36, 37–60, 61–120), `required_run_rate` in 0.5 intervals (<6.0, 6.0–6.5, ..., 16.0+), and `wickets_in_hand` (1–3, 4–6, 7–10).
* **First Innings:** Partitioned by match phase / balls remaining and `venue_par` tertiles (`Low <165`, `Medium 165–185`, `High >185`).
* Saved to [expected_tables.csv](file:///D:/RR_player_analysis/data/expected_tables.csv) (413 distinct empirical states).

### Pressure Definitions
* **Batting Pressure:** Second-innings balls with required run rate $\ge 9.0$ OR death overs (overs 16–20 / balls remaining $\le 30$) in either innings (92,767 balls).
* **Bowling Pressure:** Core chase band (RRR 8.0–12.0 with balls remaining $> 6$, 41,652 balls) and last-over component (balls remaining 1–6 across both innings, 14,358 balls). Balls in both count once exclusively under the last-over component.
* **Win-Probability Swing Benchmark:** The top 15% swings ($\ge 85\text{th percentile} \approx 0.02539$) from the second-innings Logistic Regression model provide an empirical validation benchmark.

---

## 4. Rating Formulas & Mathematical Principles (Step 6)

* **Batters:**
  $$\text{Raw Rating} = z(\text{runs\_above\_expected\_per\_100}) - z(\text{dismissal\_penalty})$$
  $$\text{Adjusted Rating} = \frac{\text{pressure\_balls} \cdot \text{Raw Rating}}{\text{pressure\_balls} + k} \quad (k=30 \text{ for RR}, k=60 \text{ for others})$$
  *(Excess dismissals impose a penalty equal to expected runs per innings in that state).*

* **Bowlers:**
  $$\text{Raw Rating} = z(\text{runs\_saved\_per\_over}) + z(\text{wicket\_component}) + 0.5 \cdot z(\text{last\_over\_component})$$
  $$\text{Adjusted Rating} = \frac{\text{pressure\_balls} \cdot \text{Raw Rating}}{\text{pressure\_balls} + k} \quad (k=24 \text{ for RR}, k=48 \text{ for others})$$
  *(Conceding runs reduces runs saved and lowers rating; wickets gained above expected are weighted by bucket runs per wicket).*

* **Internal Consistency:** Z-scores are computed independently within each ranking set using that set's qualifying mean and standard deviation.

---

## 5. Official Top 10 Rankings

### Set 1: Rajasthan Royals (RR Players Only)
#### Top 10 RR Batters (Min 30 Pressure Balls)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Karun Nair** | India | **2.9980** | +31.70 | -8.70 | 89 |
| **2** | **Vaibhav Sooryavanshi** | India | **2.1779** | +93.48 | +0.84 | 210 |
| **3** | **Rassie van der Dussen** | South Africa | **2.0826** | -6.67 | -7.26 | 217 |
| **4** | **Ravindra Jadeja** | India | **1.7956** | -12.09 | -6.29 | 720 |
| **5** | **Dhruv Jurel** | India | **1.7780** | +3.04 | -5.05 | 510 |
| **6** | **Jos Buttler** | England | **1.3316** | +25.97 | -1.67 | 922 |
| **7** | **Sanju Samson** | India | **1.0282** | +20.03 | -1.30 | 741 |
| **8** | **Rovman Powell** | West Indies | **0.9802** | +12.96 | -1.72 | 1049 |
| **9** | **Yashasvi Jaiswal** | India | **0.8976** | +32.51 | +0.11 | 579 |
| **10** | **Dasun Shanaka** | Sri Lanka | **0.7975** | +4.54 | -1.95 | 560 |

#### Top 10 RR Bowlers (Min 24 Band Balls)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **4.8247** | +1.90 | +19.50 | -0.78 | 104 |
| **2** | **FazalHaq Farooqi** | Afghanistan | **1.8657** | +1.34 | +4.52 | -4.42 | 361 |
| **3** | **Murugan Ashwin** | India | **1.6279** | +1.16 | +4.20 | 0.00 | 83 |
| **4** | **Corbin Bosch** | South Africa | **1.5925** | +0.30 | +7.96 | -1.96 | 118 |
| **5** | **Keshav Maharaj** | South Africa | **1.1992** | +0.71 | +3.11 | -20.57 | 118 |
| **6** | **Maheesh Theekshana** | Sri Lanka | **1.1770** | +0.85 | -2.35 | +9.46 | 489 |
| **7** | **Trent Boult** | New Zealand | **0.9188** | +0.86 | +0.59 | -1.47 | 570 |
| **8** | **Yuzvendra Chahal** | India | **0.8155** | +0.48 | -0.15 | +3.52 | 543 |
| **9** | **Kuldeep Sen** | India | **0.6442** | +1.23 | -0.25 | -6.01 | 131 |
| **10** | **Ravi Bishnoi** | India | **0.5785** | +0.63 | -0.03 | -1.61 | 719 |

---

### Set 2: Next Season Eligible (Appeared in 2025 & 2026)
#### Top 10 Next Season Batters (Min 30 RR / 60 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Arafat Minhas** | Pakistan | **3.8058** | +11.47 | -35.90 | 82 |
| **2** | **Vaibhav Sooryavanshi** | India | **3.4688** | +93.48 | +0.84 | 210 |
| **3** | **Atharwa Taide** | India | **3.2268** | -17.52 | -34.06 | 111 |
| **4** | **Suryakumar Yadav** | India | **2.5323** | +32.76 | -7.23 | 1177 |
| **5** | **Shan Masood** | Pakistan | **2.3082** | -14.50 | -24.46 | 119 |
| **6** | **Reeza Hendricks** | South Africa | **2.1970** | +20.89 | -9.47 | 418 |
| **7** | **Muhammad Waseem** | UAE | **2.1590** | +28.20 | -9.03 | 212 |
| **8** | **Karun Nair** | India | **2.1420** | +31.70 | -8.70 | 89 |
| **9** | **Tom Banton** | England | **2.1348** | +17.90 | -12.53 | 160 |
| **10** | **Shashank Singh** | India | **1.9810** | +25.06 | -6.80 | 451 |

#### Top 10 Next Season Bowlers (Min 24 RR / 48 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **5.9541** | +1.90 | +19.50 | -0.78 | 104 |
| **2** | **Mark Wood** | England | **2.7320** | +1.44 | +6.17 | +8.21 | 123 |
| **3** | **Mohammad Rishad Hossain** | Bangladesh | **2.4656** | +0.51 | +1.24 | +25.55 | 224 |
| **4** | **Dunith Wellalage** | Sri Lanka | **2.3764** | +1.14 | +6.35 | +4.75 | 122 |
| **5** | **Mitchell Santner** | New Zealand | **2.3751** | +1.84 | +3.14 | +6.25 | 453 |
| **6** | **Jacob Duffy** | New Zealand | **2.2070** | +0.87 | +6.62 | -2.42 | 290 |
| **7** | **Mohammad Nawaz** | Pakistan | **2.1473** | +0.64 | +6.17 | +18.41 | 305 |
| **8** | **Corbin Bosch** | South Africa | **2.0891** | +0.30 | +7.96 | -1.96 | 118 |
| **9** | **FazalHaq Farooqi** | Afghanistan | **2.0613** | +1.34 | +4.52 | -4.42 | 361 |
| **10** | **Sufiyan Muqeem** | Pakistan | **2.0062** | +2.64 | +1.01 | +0.62 | 141 |

---

### Set 3: All Players (Global Pool)
#### Top 10 All Batters (Min 30 RR / 60 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Arafat Minhas** | Pakistan | **3.8045** | +11.47 | -35.90 | 82 |
| **2** | **Vaibhav Sooryavanshi** | India | **3.4985** | +93.48 | +0.84 | 210 |
| **3** | **Atharwa Taide** | India | **3.2332** | -17.52 | -34.06 | 111 |
| **4** | **Aamir Kaleem** | Oman | **3.0889** | -31.02 | -42.74 | 67 |
| **5** | **Suryakumar Yadav** | India | **2.5313** | +32.76 | -7.23 | 1177 |
| **6** | **Shan Masood** | Pakistan | **2.3149** | -14.50 | -24.46 | 119 |
| **7** | **Matthew Wade** | Australia | **2.3118** | +7.69 | -14.45 | 303 |
| **8** | **Mohammad Rizwan** | Pakistan | **2.2398** | -4.14 | -16.03 | 497 |
| **9** | **Reeza Hendricks** | South Africa | **2.1963** | +20.89 | -9.47 | 418 |
| **10** | **Muhammad Waseem** | UAE | **2.1535** | +28.20 | -9.03 | 212 |

#### Top 10 All Bowlers (Min 24 RR / 48 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **5.8137** | +1.90 | +19.50 | -0.78 | 104 |
| **2** | **Mark Wood** | England | **2.6364** | +1.44 | +6.17 | +8.21 | 123 |
| **3** | **Mohammad Rishad Hossain** | Bangladesh | **2.3373** | +0.51 | +1.24 | +25.55 | 224 |
| **4** | **Mitchell Santner** | New Zealand | **2.3089** | +1.84 | +3.14 | +6.25 | 453 |
| **5** | **Dunith Wellalage** | Sri Lanka | **2.2935** | +1.14 | +6.35 | +4.75 | 122 |
| **6** | **Jacob Duffy** | New Zealand | **2.1338** | +0.87 | +6.62 | -2.42 | 290 |
| **7** | **Mohammad Nawaz** | Pakistan | **2.0908** | +0.64 | +6.17 | +18.41 | 305 |
| **8** | **Corbin Bosch** | South Africa | **2.0202** | +0.30 | +7.96 | -1.96 | 118 |
| **9** | **FazalHaq Farooqi** | Afghanistan | **1.9895** | +1.34 | +4.52 | -4.42 | 361 |
| **10** | **Sufiyan Muqeem** | Pakistan | **1.9480** | +2.64 | +1.01 | +0.62 | 141 |

---

## 6. Sanity Checks & Verification (Step 8)

All four validation checks passed strictly across all three sets:
1. **Batter Dismissal Derivative:** Confirmed correlation $r = -1.0000$ between dismissal penalty and raw score; holding runs scored constant, more dismissals strictly lowers a batter's rating.
2. **Bowler Economy Derivative:** Confirmed correlation $r = +1.0000$ between runs saved and raw score; holding wickets constant, conceding more runs strictly lowers a bowler's rating.
3. **Threshold Compliance:** Confirmed that **zero** non-RR players in any Top 10 have $< 60$ pressure balls (batters) or $< 48$ band balls (bowlers).
4. **Tail Integrity:** Lowest-ranked qualifying players in each set consist of tailenders with negative runs above expected and severe dismissal penalties.

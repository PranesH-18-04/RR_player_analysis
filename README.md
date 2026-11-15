# SupeRR Selector: Rajasthan Royals Pressure Analysis & Multi-Set Ranking Engine

A decision science and talent scouting system engineered for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure using game-state expected-value baselines constructed from all qualifying IPL and T20 International matches (2022 onward).

The repository features three independent, internally consistent ranking sets with tiered eligibility thresholds:
* **Set 1 (RR Players):** Franchise-specific ranking of players who appeared for Rajasthan Royals (`team_id == 3`).
* **Set 2 (Next Season Eligible):** Strategic proxy ranking of players who appeared in the most recent two seasons (`2025` and `2026`), including active RR players, with **all Pakistan players dropped** due to IPL regulatory constraints. *(Note: This serves as an objective talent availability proxy, not official squad registration).*
* **Set 3 (All Players):** Global ranking across all players in the qualifying dataset regardless of franchise or nationality, with Pakistan players fully included in the analytical pool.

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Standalone interactive dashboard (3-way pool toggle, responsive cards, RR palette)
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory analysis
│   ├── win_probability_model.py   # Second-innings logistic regression model & swing thresholding
│   └── pressure_metrics.py        # Complete pressure engine, Top-12 ICC filter, Bayesian shrinkage & rating pipeline
│
├── data/
│   ├── expected_tables.csv        # 408 empirical game-state expected value buckets
│   ├── top10_rr_batters.csv       # Top 10 Batters (Set 1: RR Only)
│   ├── top10_rr_bowlers.csv       # Top 10 Bowlers (Set 1: RR Only)
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

## 1. Top 12 ICC T20I Opponent Filter & Match Baseline (Steps 2 & 3)

### Current Top 12 ICC T20I Teams Constant
At the top of `backend/pressure_metrics.py`, the constant `TOP_12_ICC_T20I` defines the 12 teams of the official ICC Men's T20I Team Rankings:

```python
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
```

### Filtering Methodology & Match / Ball Counts
1. **IPL Matches (Unfiltered):** IPL matches are franchise competitions with no national-team opponent. All 360 resolved IPL matches (**86,713 deliveries**) are preserved unfiltered.
2. **T20I Matches (Top 12 Opponent Filter):** For T20I international matches, a delivery is kept only if the opponent is in `TOP_12_ICC_T20I`. Any T20I match where the opponent is not a top-12 team is dropped. Matches where both teams belong to the Top 12 total 397 resolved matches (**92,623 deliveries**).
3. **Dropped Matches:** 51 resolved T20I matches (**11,711 deliveries**) against non-Top 12 opponents (e.g. Nepal, UAE, Hong Kong, Oman) and 30 emerging/regional tournament matches (**6,985 deliveries**) are dropped.
4. **Unweighted:** Every delivery passing the filter carries an unweighted multiplier of **1.0** (replacing previous tiered weighting proposals).

| Match Category | Raw Matches | Resolved (Valid) Matches | Kept / Dropped Deliveries | Filter Rule Applied |
| :--- | :---: | :---: | :---: | :--- |
| **IPL Matches** | 371 | **360** | **86,713 kept** | Unfiltered franchise baseline |
| **T20I (Both Top 12)** | 428 | **397** | **92,623 kept** | Opponent confirmed in Top 12 |
| **T20I (Non-Top 12 Opponent)** | 54 | **51** | **11,711 dropped** | Dropped: opponent outside Top 12 |
| **Regional / Emerging T20** | 30 | **30** | **6,985 dropped** | Dropped: non-qualifying format |
| **TOTAL QUALIFYING BASELINE** | 883 | **757** | **179,336 balls** | **Unified Expected-Value Pool** |

### Methodological Limitations & Pakistan Opposition Status
* **Snapshot Limitation:** Applying the top 12 snapshot as of `2026-11-14` enforces a fixed quality threshold across historical matches from 2022 to 2026. While team rankings fluctuated over these 5 years, this snapshot provides an objective, reproducible cutoff that excludes associate and lower-tier fixtures without introducing arbitrary time-varying heuristics.
* **Pakistan Opposition vs. Player Eligibility:** Pakistan is ranked #7 in the ICC T20I rankings, so all T20I matches involving Pakistan against Top-12 opposition remain in the qualifying baseline. However, while Pakistan matches provide legitimate competitive opposition, individual Pakistan players are barred from participating in the IPL. Consequently, Pakistan players are excluded exclusively from **Set 2 (Next Season Eligible)**, but remain fully ranked in **Set 3 (Global All Players)**.

---

## 2. Data Audit & Confirmation (Step 1)

* **Team Confirmation:** Rajasthan Royals is confirmed dynamically from `teams.csv` and `matches.csv` as **`team_id == 3`** (`"Rajasthan Royals"`).
* **Season Confirmation:** The data spans 8 tournament seasons (`2022`, `2022/23`, `2023`, `2023/24`, `2024`, `2024/25`, `2025`, `2026`). The most recent season is confirmed as **`2026`**, preceded by **`2025`**.
* **Nationality Mapping:** Mapped across all 1,217 players in `competition_team_player_mapping.csv`, correctly identifying domestic Indian players and overseas professionals.

---

## 3. Three Ranking Sets & Tiered Thresholds (Steps 3 & 7)

### Definitions of the Sets
1. **Set 1 (RR Players):** Every player who appeared for Rajasthan Royals (`team_id == 3`) in the qualifying dataset.
2. **Set 2 (Next Season Eligible):** Every player who appeared in the most recent two seasons (`2025` and `2026`) in the qualifying data, **with all Pakistan players removed**. *(Serves as a realistic talent availability proxy for IPL auction planning).*
3. **Set 3 (All Players):** Every player in the qualifying dataset across all teams and nationalities, **including Pakistan players**.

### Tiered Minimum Sample Thresholds
* **RR Players (Set 1, and RR players in Sets 2 & 3):**
  * Batters: $\ge \mathbf{30}$ pressure deliveries faced.
  * Bowlers: $\ge \mathbf{24}$ deliveries in the RRR 8–12 chase band.
  * Bayesian Shrinkage: $k = 30$ (batters), $k = 24$ (bowlers).
* **Non-RR Players (Outside RR in Sets 2 & 3):**
  * Batters: raised to $\ge \mathbf{60}$ pressure deliveries faced.
  * Bowlers: raised to $\ge \mathbf{48}$ deliveries in the RRR 8–12 chase band.
  * Bayesian Shrinkage: $k = 60$ (batters), $k = 48$ (bowlers).

### Threshold Removal & Eligibility Audit
| Ranking Set | Player Pool | Role | Evaluated Players | Qualified Players | Removed by Threshold |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Set 1: RR Players** | **69** | Batters | 57 | **33** | 24 |
| | | Bowlers | 45 | **34** | 11 |
| **Set 2: Next Season Eligible (ex-Pak)** | **454** | Batters | 373 | **184** | 189 |
| | | Bowlers | 284 | **126** | 158 |
| **Set 3: All Players (Global with Pak)** | **705** | Batters | 545 | **220** | 325 |
| | | Bowlers | 398 | **150** | 248 |

---

## 4. Expected-Value Baselines & Pressure Definitions (Steps 4 & 5)

A single unified expected-value baseline was constructed across all **179,336 qualifying deliveries** in 757 qualifying matches:
* **Second Innings (Chase):** Partitioned by `balls_remaining` (1–6, 7–12, 13–24, 25–36, 37–60, 61–120), `required_run_rate` in 0.5 intervals (<6.0, 6.0–6.5, ..., 16.0+), and `wickets_in_hand` (1–3, 4–6, 7–10).
* **First Innings:** Partitioned by match phase / balls remaining and `venue_par` tertiles (`Low <165`, `Medium 165–185`, `High >185`).
* Saved to [expected_tables.csv](file:///D:/RR_player_analysis/data/expected_tables.csv) (408 distinct empirical states).

### Pressure Definitions
* **Batting Pressure:** Second-innings deliveries with required run rate $\ge 9.0$ OR death overs (overs 16–20 / balls remaining $\le 30$) in either innings.
* **Bowling Pressure:** Core chase band (RRR 8.0–12.0 with balls remaining $> 6$) and last-over component (overs 20 / balls remaining 1–6 across both innings). Balls in both count once exclusively under the last-over component.

---

## 5. Rating Formulas & Mathematical Principles (Step 6)

* **Batters:**
  $$\text{Raw Rating} = z(\text{runs\_above\_expected\_per\_100}) - z(\text{dismissal\_penalty})$$
  $$\text{Adjusted Rating} = \frac{\text{pressure\_balls} \cdot \text{Raw Rating}}{\text{pressure\_balls} + k} \quad (k=30 \text{ for RR}, k=60 \text{ for others})$$
  *(Excess dismissals impose a penalty equal to expected runs per innings in that state).*

* **Bowlers:**
  $$\text{Raw Rating} = z(\text{runs\_saved\_per\_over}) + z(\text{wicket\_component}) + 0.5 \cdot z(\text{last\_over\_component})$$
  $$\text{Adjusted Rating} = \frac{\text{pressure\_balls} \cdot \text{Raw Rating}}{\text{pressure\_balls} + k} \quad (k=24 \text{ for RR}, k=48 \text{ for others})$$
  *(Conceding runs reduces runs saved and strictly lowers rating; wickets gained above expected are weighted by bucket runs per wicket).*

* **Internal Standardization:** Z-scores are computed independently within each ranking set using that set's qualifying mean and standard deviation.

---

## 6. Official Top 10 Rankings

### Set 1: Rajasthan Royals (RR Players Only)
#### Top 10 RR Batters (Min 30 Pressure Balls)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Karun Nair** | India | **2.9310** | +30.07 | -8.41 | 89 |
| **2** | **Rassie van der Dussen** | South Africa | **2.0651** | -7.68 | -7.13 | 217 |
| **3** | **Vaibhav Sooryavanshi** | India | **1.8440** | +86.85 | 1.44 | 200 |
| **4** | **Ravindra Jadeja** | India | **1.8231** | -13.28 | -6.32 | 720 |
| **5** | **Dhruv Jurel** | India | **1.7217** | +1.22 | -4.89 | 510 |
| **6** | **Jos Buttler** | England | **1.2993** | +25.17 | -1.52 | 922 |
| **7** | **Sanju Samson** | India | **1.0803** | +19.28 | -1.40 | 726 |
| **8** | **Rovman Powell** | West Indies | **0.9688** | +12.07 | -1.64 | 1049 |
| **9** | **Yashasvi Jaiswal** | India | **0.7922** | +31.38 | 0.44 | 579 |
| **10** | **Dasun Shanaka** | Sri Lanka | **0.7905** | +3.32 | -1.91 | 557 |

#### Top 10 RR Bowlers (Min 24 Band Balls)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **4.8593** | +2.01 | +20.23 | -0.79 | 104 |
| **2** | **Murugan Ashwin** | India | **1.6210** | +1.26 | +4.70 | +0.00 | 83 |
| **3** | **FazalHaq Farooqi** | Afghanistan | **1.5151** | +1.44 | +3.67 | -5.37 | 327 |
| **4** | **Keshav Maharaj** | South Africa | **1.1458** | +0.85 | +3.11 | -20.57 | 118 |
| **5** | **Maheesh Theekshana** | Sri Lanka | **1.0756** | +0.93 | -2.29 | +10.94 | 483 |
| **6** | **Jimmy Neesham** | New Zealand | **1.0646** | +0.29 | +4.40 | +3.01 | 88 |
| **7** | **Trent Boult** | New Zealand | **0.7987** | +0.94 | +0.60 | -1.23 | 570 |
| **8** | **Yuzvendra Chahal** | India | **0.6218** | +0.47 | +0.11 | +3.45 | 535 |
| **9** | **Kuldeep Sen** | India | **0.5290** | +1.28 | -0.13 | -6.08 | 131 |
| **10** | **Ravi Bishnoi** | India | **0.4764** | +0.72 | +0.08 | -1.61 | 712 |

---

### Set 2: Next Season Eligible (Appeared in 2025 & 2026 - Pakistan Excluded)
#### Top 10 Next Season Batters (Min 30 RR / 60 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Atharwa Taide** | India | **4.6578** | -19.81 | -34.54 | 111 |
| **2** | **Vaibhav Sooryavanshi** | India | **3.0983** | +86.85 | 1.44 | 200 |
| **3** | **Suryakumar Yadav** | India | **2.7301** | +30.03 | -6.49 | 1154 |
| **4** | **Reeza Hendricks** | South Africa | **2.6511** | +19.79 | -9.14 | 418 |
| **5** | **Tom Banton** | England | **2.6482** | +16.45 | -12.40 | 160 |
| **6** | **Karun Nair** | India | **2.4836** | +30.07 | -8.41 | 89 |
| **7** | **Shashank Singh** | India | **2.3445** | +23.74 | -6.77 | 451 |
| **8** | **Namboori Tilak Varma** | India | **2.1532** | +10.19 | -7.67 | 1199 |
| **9** | **Sai Sudharsan** | India | **2.1270** | +12.68 | -7.39 | 709 |
| **10** | **Lokesh Rahul** | India | **1.9920** | +8.81 | -7.44 | 777 |

#### Top 10 Next Season Bowlers (Min 24 RR / 48 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **6.1722** | +2.01 | +20.23 | -0.79 | 104 |
| **2** | **Mark Wood** | England | **2.7687** | +1.52 | +6.18 | +7.91 | 123 |
| **3** | **Mahedi Hasan** | Bangladesh | **2.7036** | +1.80 | +6.43 | -9.00 | 150 |
| **4** | **Mitchell Santner** | New Zealand | **2.4134** | +1.75 | +3.53 | +6.25 | 402 |
| **5** | **Jacob Duffy** | New Zealand | **2.3299** | +0.94 | +6.74 | -2.45 | 290 |
| **6** | **Mohammad Rishad Hossain** | Bangladesh | **2.1884** | +0.47 | +1.51 | +22.87 | 182 |
| **7** | **Corbin Bosch** | South Africa | **1.9641** | +0.42 | +8.34 | -1.93 | 118 |
| **8** | **Nathan Ellis** | Australia | **1.9194** | +1.34 | +2.28 | +3.04 | 314 |
| **9** | **FazalHaq Farooqi** | Afghanistan | **1.9091** | +1.44 | +3.67 | -5.37 | 327 |
| **10** | **Axar Patel** | India | **1.8904** | +1.02 | +3.70 | -0.80 | 580 |

---

### Set 3: All Players (Global Pool - Pakistan Included)
#### Top 10 All Batters (Min 30 RR / 60 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Atharwa Taide** | India | **4.2893** | -19.81 | -34.54 | 111 |
| **2** | **Vaibhav Sooryavanshi** | India | **3.2597** | +86.85 | 1.44 | 200 |
| **3** | **Shan Masood** | Pakistan | **3.0393** | -15.93 | -24.30 | 119 |
| **4** | **Matthew Wade** | Australia | **2.9089** | +6.45 | -14.50 | 303 |
| **5** | **Mohammad Rizwan** | Pakistan | **2.8155** | -5.69 | -15.60 | 487 |
| **6** | **Suryakumar Yadav** | India | **2.7004** | +30.03 | -6.49 | 1154 |
| **7** | **Shakib Al Hasan** | Bangladesh | **2.6248** | -2.65 | -16.00 | 206 |
| **8** | **Reeza Hendricks** | South Africa | **2.5721** | +19.79 | -9.14 | 418 |
| **9** | **Tom Banton** | England | **2.5399** | +16.45 | -12.40 | 160 |
| **10** | **Karun Nair** | India | **2.4366** | +30.07 | -8.41 | 89 |

#### Top 10 All Bowlers (Min 24 RR / 48 Others)
| Rank | Player | Nationality | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | India | **5.9868** | +2.01 | +20.23 | -0.79 | 104 |
| **2** | **Mark Wood** | England | **2.6393** | +1.52 | +6.18 | +7.91 | 123 |
| **3** | **Mahedi Hasan** | Bangladesh | **2.6010** | +1.80 | +6.43 | -9.00 | 150 |
| **4** | **Mohammad Nawaz** | Pakistan | **2.4118** | +0.79 | +7.50 | +20.64 | 250 |
| **5** | **Mitchell Santner** | New Zealand | **2.3037** | +1.75 | +3.53 | +6.25 | 402 |
| **6** | **Jacob Duffy** | New Zealand | **2.1941** | +0.94 | +6.74 | -2.45 | 290 |
| **7** | **Mohammad Rishad Hossain** | Bangladesh | **2.0494** | +0.47 | +1.51 | +22.87 | 182 |
| **8** | **Murugan Ashwin** | India | **1.9316** | +1.26 | +4.70 | +0.00 | 83 |
| **9** | **Corbin Bosch** | South Africa | **1.8499** | +0.42 | +8.34 | -1.93 | 118 |
| **10** | **Nathan Ellis** | Australia | **1.7864** | +1.34 | +2.28 | +3.04 | 314 |

---

## 7. Sanity Checks & Verification (Step 9)

All sanity checks passed strictly across all three sets:
1. **Batter Dismissal Derivative:** Confirmed correlation $r = -1.0000$ between dismissal penalty and raw score; holding runs scored constant, more dismissals strictly lowers a batter's rating.
2. **Bowler Economy Derivative:** Confirmed correlation $r = +1.0000$ between runs saved and raw score; holding wickets constant, conceding more runs strictly lowers a bowler's rating.
3. **Pakistan Exclusion Verification:** Verified **zero (0) Pakistan players** in Set 2 (Next Season Eligible). In Set 3, Pakistan players (such as Shan Masood #3, Mohammad Rizwan #5, Mohammad Nawaz #4) are legitimately ranked.
4. **Threshold Compliance:** Confirmed that **zero** non-RR players in any Top 10 have $< 60$ pressure balls (batters) or $< 48$ band balls (bowlers).

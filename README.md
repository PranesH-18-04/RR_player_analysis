# SupeRR Selector: Rajasthan Royals Pressure Analysis & Multi-Tier Analytics Framework

A decision science, pressure analytics, and talent scouting engine engineered for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure using game-state expected-value baselines constructed from qualifying IPL, T20 International, Development, and Youth fixtures (2022 onward).

The framework strictly partitions matches into three tiers to eliminate quality cross-contamination:
* **SENIOR:** Full national teams (with official ICC Top-12 opponent filter) and Indian Premier League (IPL) franchise matches.
  * **Set 1 (RR Players):** Franchise-specific ranking of players who appeared for Rajasthan Royals (`team_id == 3`).
  * **Set 2 (Next Season Eligible):** Strategic proxy ranking of players who appeared in the most recent two seasons (`2025` and `2026`), with **all Pakistan players excluded** per IPL auction eligibility constraints.
  * **Set 3 (All Senior Players):** Global ranking across all senior players in qualifying matches, with Pakistan players fully included.
* **DEVELOPMENT:** A-teams and emerging competitions (e.g., *Mens T20 Emerging Teams Asia Cup 2024*, Comp 2670).
  * **Development Players Sub-List:** Ranked exclusively on Development tier deliveries.
* **YOUTH:** Under-age and youth tournament fixtures (e.g., *Asia Cup Rising Stars 2025*, Comp 3289).
  * **Young Players Sub-List:** Ranked exclusively on Youth tier deliveries.

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Standalone interactive dashboard (4-way pool toggle, sub-toggle, RR palette)
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory analysis
│   ├── win_probability_model.py   # Second-innings logistic regression model & swing thresholding
│   └── pressure_metrics.py        # Complete multi-tier engine, TEAM_TIER mapping, ICC filter & shrinkage rating pipeline
│
├── data/
│   ├── expected_tables.csv        # 408 empirical game-state expected value buckets (Senior pool)
│   ├── expected_tables_development.csv # 217 empirical game-state buckets (Development & Youth pool)
│   ├── top10_rr_batters.csv       # Top 10 Batters (Set 1: RR Only)
│   ├── top10_rr_bowlers.csv       # Top 10 Bowlers (Set 1: RR Only)
│   ├── top10_next_season_batters.csv # Top 10 Batters (Set 2: Next Season Eligible)
│   ├── top10_next_season_bowlers.csv # Top 10 Bowlers (Set 2: Next Season Eligible)
│   ├── top10_all_batters.csv      # Top 10 Batters (Set 3: All Senior Players - Global)
│   ├── top10_all_bowlers.csv      # Top 10 Bowlers (Set 3: All Senior Players - Global)
│   ├── top10_young_batters.csv    # Top 10 Young Batters (Youth Comp 3289)
│   ├── top10_young_bowlers.csv    # Top Young Bowlers (Youth Comp 3289)
│   ├── top10_development_batters.csv # Top 10 Development Batters (Emerging Comp 2670)
│   ├── top10_development_bowlers.csv # Top Development Bowlers (Emerging Comp 2670)
│   ├── batters_rr.csv             # Full per-player metrics for RR batters
│   ├── bowlers_rr.csv             # Full per-player metrics for RR bowlers
│   ├── batters_next_season.csv    # Full per-player metrics for Next Season batters
│   ├── bowlers_next_season.csv    # Full per-player metrics for Next Season bowlers
│   ├── batters_all.csv            # Full per-player metrics for all global senior batters
│   ├── bowlers_all.csv            # Full per-player metrics for all global senior bowlers
│   ├── batters_young.csv          # Full per-player metrics for Young batters (Youth)
│   ├── bowlers_young.csv          # Full per-player metrics for Young bowlers (Youth)
│   ├── batters_development.csv    # Full per-player metrics for Development batters (A-Teams)
│   ├── bowlers_development.csv    # Full per-player metrics for Development bowlers (A-Teams)
│   ├── batters_pressure.csv       # Backward-compatible RR batters table
│   ├── bowlers_pressure.csv       # Backward-compatible RR bowlers table
│   └── win_prob_curves.png        # In-play win probability validation curves
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## 1. Team Tier Classification & Match Separation (Steps 2 & 3)

### `TEAM_TIER` Mapping Constant
At the top of `backend/pressure_metrics.py`, every team appearing across the dataset is deterministically mapped to one of three tiers without guessing:
* **`SENIOR`:** Full national cricket teams (23 teams) and IPL franchises (10 franchises).
* **`DEVELOPMENT`:** "A", "B", "C", "D", U23, Emerging, Academy, and XI sides (e.g., `India A`, `Pakistan A`, `Sri Lanka A`, `Afghanistan A`, `Bangladesh A`).
* **`YOUTH`:** Teams labeled U16, U17, U19, U21, or participating in designated youth tournaments.

```python
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
    "Afghanistan": "SENIOR", "Australia": "SENIOR", "Bangladesh": "SENIOR",
    "Canada": "SENIOR", "England": "SENIOR", "Hong Kong": "SENIOR",
    "India": "SENIOR", "Ireland": "SENIOR", "Japan": "SENIOR",
    "Kuwait": "SENIOR", "Nepal": "SENIOR", "Netherlands": "SENIOR",
    "New Zealand": "SENIOR", "Oman": "SENIOR", "Pakistan": "SENIOR",
    "Scotland": "SENIOR", "Singapore": "SENIOR", "South Africa": "SENIOR",
    "Sri Lanka": "SENIOR", "United Arab Emirates": "SENIOR",
    "United States Of America": "SENIOR", "West Indies": "SENIOR",
    "Zimbabwe": "SENIOR",

    # Development / "A" Sides (DEVELOPMENT)
    "Afghanistan A": "DEVELOPMENT", "Bangladesh A": "DEVELOPMENT",
    "India A": "DEVELOPMENT", "Pakistan A": "DEVELOPMENT",
    "Sri Lanka A": "DEVELOPMENT",
    "India B": "DEVELOPMENT", "India C": "DEVELOPMENT",
    "India D": "DEVELOPMENT", "India U23": "DEVELOPMENT",

    # Youth Sides (YOUTH)
    "India U21": "YOUTH", "India U19": "YOUTH",
    "India U17": "YOUTH", "India U16": "YOUTH"
}
```

### Complete Breakdown of Matches and Deliveries by Tier

| Match Tier | Competitions Included | Raw Matches | Resolved Matches | Raw Deliveries | Valid Deliveries | Policy Applied |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **SENIOR** | IPL (2022–2026), T20I (2022–2026) | **853** | **757** | **194,014** | **179,336** | Top 12 T20I opponent filter applied to international matches (51 T20I dropped). IPL unfiltered. |
| **DEVELOPMENT** | *Mens T20 Emerging Teams Asia Cup 2024* (Comp 2670) | **15** | **15** | **3,519** | **3,519** | Kept in full; opponent filter not applied (distinct competition tier). |
| **YOUTH** | *Asia Cup Rising Stars 2025* (Comp 3289) | **15** | **15** | **3,466** | **3,466** | Kept in full; opponent filter not applied (distinct youth tier). |
| **TOTAL DATASET** | All 8 tournament series | **883** | **787** | **200,999** | **186,321** | **Strict tier separation (Zero cross-tier leakage)** |

---

## 2. Separate Expected-Value Baselines (Step 4)

### Senior Baseline (`data/expected_tables.csv`)
* **Size:** 408 empirical game-state buckets constructed from **179,336 deliveries** (757 senior matches).
* **Usage:** Exclusively used to evaluate Senior ranking sets (Set 1 RR, Set 2 Next Season Eligible, Set 3 All Senior Players).
* **Opponent Standard:** All international matches strictly against official ICC Top-12 opponents.

### Lower-Tier Baseline (`data/expected_tables_development.csv`)
* **Size:** 217 empirical game-state buckets constructed across all **6,985 deliveries** in the 30 lower-tier matches (15 Development + 15 Youth matches).
* **Sample Reliability Threshold:** Due to smaller aggregate volume, buckets with $< 50$ balls are marked as `Unreliable (<50 balls)` (196 buckets), while buckets with $\ge 50$ balls are marked as `Reliable (>=50 balls)` (21 buckets).
* **Strict Non-Comparability Notice:** Lower-tier ratings are computed against this separate 217-bucket baseline and standardized internally within their respective sub-list. They are **not directly comparable** to Senior ratings due to substantial differences in competition quality, game dynamics, and baseline expectations.

---

## 3. Multi-Tier Player Tracking & Atharva Taide Case Study

A total of **149 players** appeared in more than one competition tier. To maintain mathematical integrity:
1. **Zero Ball Contamination:** Deliveries faced or bowled in Development/Youth fixtures are completely excluded from senior sets (Sets 1, 2, and 3).
2. **Independent Tier Entries:** When a player appears in multiple tiers, they are evaluated separately in each tier using only the deliveries from that specific tier.

### Multi-Tier Player Highlights
| Player | Tiers Appeared | Senior Matches | Development Matches | Youth Matches | Total Matches |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Namboori Tilak Varma** | Senior, Development | 138 | 4 | 0 | 142 |
| **Abhishek Sharma** | Senior, Development | 134 | 4 | 0 | 138 |
| **Jitesh Sharma** | Senior, Youth | 107 | 0 | 4 | 111 |
| **Nehal Wadhera** | Senior, Development, Youth | 60 | 4 | 4 | 68 |
| **Ramandeep Singh** | Senior, Development, Youth | 56 | 4 | 4 | 64 |
| **Vaibhav Sooryavanshi** | Senior, Youth | 54 | 0 | 4 | 58 |
| **Sediqullah Atal** | Senior, Development | 31 | 5 | 0 | 36 |

### Atharwa Taide Analytical Profile
In Senior cricket (Sets 2 & 3), Atharwa Taide ranks #1 because across 111 pressure deliveries in 4 innings, he suffered only 3 dismissals against 6.27 expected (excess dismissals: $-3.27$, yielding a massive negative dismissal penalty of $-34.54$). In Development matches, his volume did not cross the qualification cutoff, so he is classified as `Ineligible (<30 balls)` in the lower-tier tables. This highlights the scout utility of tiered evaluation: distinguishing high-pressure senior franchise execution from developmental tournament exposure.

---

## 4. Young & Development Rankings (Step 4, 5, 6)

### Eligibility Thresholds for Young & Development Sub-Lists
* **Batters:** Min **30** pressure balls in that tier ($k = 30.0$).
* **Bowlers:** Min **24** band balls (RRR 8.0–12.0) in that tier ($k = 24.0$).
* Players with fewer balls are retained in the full per-player rosters (`batters_young.csv`, `bowlers_young.csv`, `batters_development.csv`, `bowlers_development.csv`) with clear `Ineligible` badges and sample notes.

---

### Young Players Sub-List (Comp 3289: Asia Cup Rising Stars 2025)

#### Top 10 Young Batters (Youth Matches Only)
| Rank | Player | Team | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls | Sample Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Sohaib Khan** | United Arab Emirates | **2.1852** | +39.02 | -111.94 | 43 | Qualified (43 >= 30 balls across 2 inns) |
| **2** | **Jitesh Sharma** | India A | **0.9785** | +59.37 | -25.11 | 49 | Qualified (49 >= 30 balls across 4 inns) |
| **3** | **SM Meherob Hasan** | Bangladesh A | **0.4186** | +33.34 | -20.15 | 38 | Qualified (38 >= 30 balls across 4 inns) |
| **4** | **Milan Rathnayake** | Sri Lanka A | **0.4017** | +54.30 | -0.40 | 42 | Qualified (42 >= 30 balls across 3 inns) |
| **5** | **Haider Shah** | United Arab Emirates | **-0.0045** | +28.14 | +0.17 | 31 | Qualified (31 >= 30 balls across 2 inns) |
| **6** | **Aryan Bisht** | Oman | **-0.3488** | +2.10 | +0.56 | 30 | Qualified (30 >= 30 balls across 2 inns) |
| **7** | **Muhammad Arfan** | United Arab Emirates | **-0.6883** | -41.18 | -16.63 | 34 | Qualified (34 >= 30 balls across 3 inns) |
| **8** | **Wasim Ali** | Oman | **-0.7720** | -31.10 | -0.79 | 31 | Qualified (31 >= 30 balls across 3 inns) |
| **9** | **Nehal Wadhera** | India A | **-0.8916** | -29.01 | -0.92 | 47 | Qualified (47 >= 30 balls across 4 inns) |
| **10** | **Qais Ahmad** | Afghanistan A | **-0.9388** | -43.64 | -4.10 | 35 | Qualified (35 >= 30 balls across 3 inns) |

#### Young Bowlers (Youth Matches Only - All Qualified)
*Note: Across the 15 Youth matches, exactly 2 bowlers met the 24 band balls qualification threshold. All 70 participating bowlers are preserved in `bowlers_young.csv`.*
| Rank | Player | Team | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls | Sample Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Garuka Sanketh** | Sri Lanka A | **1.3684** | -0.24 | -0.29 | -5.87 | 52 | Qualified (29 band balls, 23 last-over balls) |
| **2** | **SM Meherob Hasan** | Bangladesh A | **-1.0204** | -1.70 | -5.61 | +0.00 | 25 | Qualified (25 band balls, 0 last-over balls) |

---

### Development Players Sub-List (Comp 2670: Mens T20 Emerging Teams Asia Cup 2024)

#### Top 10 Development Batters (Development Matches Only)
| Rank | Player | Team | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls | Sample Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Sediqullah Atal** | Afghanistan A | **1.0472** | +42.28 | -9.94 | 65 | Robust sample (65 balls across 5 inns) |
| **2** | **Arafat Minhas** | Pakistan A | **0.9849** | +17.45 | -28.70 | 47 | Qualified (47 >= 30 balls across 4 inns) |
| **3** | **Ramandeep Singh** | India A | **0.8386** | +40.03 | -8.66 | 48 | Qualified (48 >= 30 balls across 4 inns) |
| **4** | **Jatinder Singh** | Oman | **0.6561** | -38.05 | -61.99 | 31 | Qualified (31 >= 30 balls across 2 inns) |
| **5** | **Sahan Arachchige** | Sri Lanka A | **0.4646** | +24.87 | -8.45 | 53 | Qualified (53 >= 30 balls across 4 inns) |
| **6** | **Babar Hayat** | Hong Kong | **0.2202** | +17.94 | -6.42 | 64 | Robust sample (64 balls across 3 inns) |
| **7** | **Namboori Tilak Varma** | India A | **-0.1472** | +9.44 | -1.85 | 30 | Qualified (30 >= 30 balls across 4 inns) |
| **8** | **Haider Ali** | Pakistan A | **-0.3048** | -20.04 | -17.56 | 33 | Qualified (33 >= 30 balls across 4 inns) |
| **9** | **Hammad Mirza** | Oman | **-0.3234** | +4.60 | -0.51 | 39 | Qualified (39 >= 30 balls across 3 inns) |
| **10** | **Yasim Murtaza** | Hong Kong | **-0.4420** | -5.40 | -4.34 | 42 | Qualified (42 >= 30 balls across 3 inns) |

#### Top Development Bowlers (Development Matches Only - All Qualified)
*Note: Exactly 8 bowlers in the 15 Emerging Teams matches met the 24 band balls qualification threshold. All 72 participating bowlers are preserved in `bowlers_development.csv`.*
| Rank | Player | Team | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls | Sample Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Shahnawaz Dhani** | Pakistan A | **1.1476** | +4.35 | +1.46 | +0.00 | 36 | Qualified (36 band balls, 0 last-over balls) |
| **2** | **Eshan Malinga** | Sri Lanka A | **0.8826** | +2.01 | +4.74 | +1.59 | 42 | Qualified (36 band balls, 6 last-over balls) |
| **3** | **Allah Ghazanfar** | Afghanistan A | **0.8732** | +1.37 | +6.53 | -4.00 | 39 | Qualified (33 band balls, 6 last-over balls) |
| **4** | **Faridoon Dawoodzai** | Afghanistan A | **0.3804** | -1.47 | +9.94 | -7.50 | 32 | Qualified (26 band balls, 6 last-over balls) |
| **5** | **Nimesh Vimukthi** | Sri Lanka A | **-0.1095** | +0.31 | +0.84 | +0.00 | 42 | Qualified (42 band balls, 0 last-over balls) |
| **6** | **Sharafuddin Ashraf** | Afghanistan A | **-0.1134** | +0.83 | -0.64 | +0.00 | 33 | Qualified (33 band balls, 0 last-over balls) |
| **7** | **Karim Janat** | Afghanistan A | **-1.1437** | -1.18 | -4.78 | -4.50 | 34 | Qualified (28 band balls, 6 last-over balls) |
| **8** | **Bilal Sami** | Afghanistan A | **-1.8134** | -2.50 | -7.25 | +0.09 | 35 | Qualified (29 band balls, 6 last-over balls) |

---

## 5. Senior Rankings: Sets 1, 2, and 3

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

### Set 3: All Senior Players (Global Senior Pool - Pakistan Included)
#### Top 10 All Senior Batters (Min 30 RR / 60 Others)
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

#### Top 10 All Senior Bowlers (Min 24 RR / 48 Others)
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

## 6. Sanity Checks & Mathematical Verification (Step 7)

Every output produced by `backend/pressure_metrics.py` passes automated mathematical assertions:
1. **Zero Ball Leakage:** Zero Development or Youth deliveries appear in Senior outputs; zero Senior deliveries appear in Young or Development outputs. Match ID sets are strictly disjoint.
2. **Batter Dismissal Derivative ($r = -1.000$):** Holding runs scored constant, incurring additional dismissals strictly penalizes the batter rating. Correlation between $z(\text{dismissal})$ and the negative component of the raw score is exactly $-1.0000$.
3. **Bowler Economy Derivative ($r = +1.000$):** Holding wickets and last-over score constant, conceding additional runs strictly lowers runs saved and bowler rating. Correlation is exactly $+1.0000$.
4. **Threshold Adherence:** Every player in the Young and Development Top-10 lists strictly meets their respective minimum volume threshold ($\ge 30$ pressure balls for batters; $\ge 24$ band balls for bowlers).
5. **Pakistan Exclusion Verification:** Confirmed exactly 0 Pakistan players in Set 2 (Next Season Eligible). In Set 3, Pakistan players (Shan Masood #3, Mohammad Rizwan #5, Mohammad Nawaz #4) are legitimately ranked.

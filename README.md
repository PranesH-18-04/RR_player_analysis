# SupeRR Selector: Rajasthan Royals Pressure Analysis & Multi-Tier Analytics Framework

A production decision-science, pressure analytics, and player-acquisition framework engineered for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure using game-state expected-value baselines constructed from qualifying IPL, T20 International, Development, and Youth fixtures (2022 onward).

The framework strictly partitions matches into three tiers to eliminate quality cross-contamination:
* **SENIOR:** Full national teams (with official ICC Top-12 opponent filter) and Indian Premier League (IPL) franchise matches.
  * **Set 1 (RR Players):** Franchise-specific ranking of players who actually batted or bowled for Rajasthan Royals (`team_id == 3`) in senior matches.
  * **Set 2 (Next Season Eligible):** Strategic proxy ranking of players who appeared in the most recent two seasons (`2025` and `2026`), with **all Pakistan players excluded** per IPL auction eligibility constraints.
  * **Set 3 (All Senior Players):** Global ranking across all senior players in qualifying matches, with Pakistan players fully included.
* **DEVELOPMENT:** A-teams and emerging competitions (e.g., *Mens T20 Emerging Teams Asia Cup 2024*, Comp 2670).
  * **Development Players Sub-List:** Ranked exclusively on Development tier deliveries.
* **YOUTH:** Under-age and youth tournament fixtures (e.g., *Asia Cup Rising Stars 2025*, Comp 3289).
  * **Young Players Sub-List:** Ranked exclusively on Youth tier deliveries.

---

## Key Algorithmic Advancements (Production Engine 2.0)

1. **Per-Innings Pressure Value (Change 1):**
   Batters are evaluated via a unified per-innings outcome in runs:
   $$\text{innings\_value} = (\text{actual\_runs} - \text{expected\_runs}) - (\text{actual\_dismissals} - \text{expected\_dismissals}) \times \text{expected\_runs\_per\_innings}$$
   Batter pressure rating is the mean of $\text{innings\_value}$ divided by the standard deviation of innings value across all qualifying innings in the set ($z$-like scale). All 4,873 senior pressure innings are exported to `data/batter_pressure_innings.csv`.

2. **Required-Run-Rate Chase Tracking (Change 2):**
   For every chase delivery, the batter's cumulative scoring rate (runs/ball) is evaluated against the required run rate ($RRR / 6.0$). The metric `pct_balls_below_rrr` captures the fraction of pressure balls spent lagging the target rate. This enters as an explicit penalty:
   $$\text{raw\_rating} = z(\text{innings\_value})_{\text{clipped}} - 0.5 \times z(\text{pct\_balls\_below\_rrr})_{\text{clipped}}$$
   governed by named constant `RRR_PENALTY_WEIGHT = 0.5`.

3. **Strict RR Match Participation Mapping (Change 3):**
   Replaced squad registration mapping with match participation filter (`batting_team_id == 3` or `bowling_team_id == 3` in senior matches). Exactly 13 unparticipating squad players were removed (pool corrected from 69 to 56 players).

4. **Corrected Pressure Definition (Change 4):**
   * **Batting Pressure:** Second innings only, where $RRR \ge 9.0$ or death overs (overs 16–20) are bowled. All 26,421 first-innings balls are completely excluded from batting pressure.
   * **Bowling Pressure:** Second innings with $RRR \in [8.0, 12.0]$, plus the last over ($\le 6$ balls remaining) of either innings as a separate component.

5. **Striker Dismissal Attribution (Change 5):**
   Dismissals are strictly mapped to `is_wicket & (dismissed_player_id == striker_id)`. Non-striker run outs and retirements (e.g., Match 40709 Tilak Varma retired out while Hardik Pandya was striker) are explicitly handled as non-dismissals for the striker.

6. **Leave-One-Out (LOO) Baseline (Change 6):**
   To prevent self-leakage, every ranked player is evaluated against an algebraic Leave-One-Out baseline that subtracts their own balls, runs, and dismissals from the empirical game-state buckets. Average absolute difference across senior deliveries is 0.007128 runs/ball (0.018217 runs/ball per bucket).

7. **Innings Floor & Component Clipping (Change 7):**
   * `MIN_INNINGS_BATTER = 8` and `MIN_INNINGS_BOWLER = 8` for Senior sets.
   * `MIN_INNINGS_DEV_YOUTH = 2` for 5-match developmental/youth tournaments.
   * `Z_CLIP = 2.0`: Every $z$-score component is clipped to $[-2.0, +2.0]$ prior to weighting, with unclipped values preserved in the output tables.

8. **Synthetic Ranking-Order Sanity Test Suite (Change 8):**
   Replaced circular correlation checks with deterministic synthetic ranking-order assertions:
   * *(a)* Same runs + more dismissals ranks strictly lower ($2.0 < 20.0$).
   * *(b)* Conceding more runs in the same bowling band ranks strictly lower ($-1.0 < +2.0$).
   * *(c)* Scoring below RRR in $>50\%$ of balls ranks strictly lower ($0.333 < 2.000$).

9. **Dynamic Two Most Recent Seasons (Change 9):**
   Set 2 dynamically selects `all_seasons[-2:]` (`['2025', '2026']`), capturing 454 eligible active players.

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Interactive dashboard with 5 pool tabs, dynamic Chart.js & full roster tables
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory analysis
│   ├── win_probability_model.py   # Second-innings logistic regression model & swing thresholding
│   └── pressure_metrics.py        # Production Engine 2.0: Multi-tier LOO pipeline, RRR penalty & synthetic tests
│
├── data/
│   ├── batter_pressure_innings.csv # Change 1: Complete 4,873 pressure innings breakdown
│   ├── expected_tables.csv        # 413 empirical game-state buckets (Senior pool)
│   ├── expected_tables_development.csv # 217 empirical game-state buckets (Development & Youth pool)
│   ├── top10_rr_batters.csv       # Top 10 Batters (Set 1: RR Only)
│   ├── top10_rr_bowlers.csv       # Top 10 Bowlers (Set 1: RR Only)
│   ├── top10_next_season_batters.csv # Top 10 Batters (Set 2: Next Season Eligible)
│   ├── top10_next_season_bowlers.csv # Top 10 Bowlers (Set 2: Next Season Eligible)
│   ├── top10_all_batters.csv      # Top 10 Batters (Set 3: All Senior Players - Global)
│   ├── top10_all_bowlers.csv      # Top 10 Bowlers (Set 3: All Senior Players - Global)
│   ├── top10_development_batters.csv # Top 10 Development Batters (Comp 2670)
│   ├── top10_development_bowlers.csv # Top 10 Development Bowlers (Comp 2670)
│   ├── top10_young_batters.csv    # Top 10 Young Batters (Comp 3289)
│   ├── top10_young_bowlers.csv    # Top 10 Young Bowlers (Comp 3289)
│   ├── batters_rr.csv / bowlers_rr.csv
│   ├── batters_next_season.csv / bowlers_next_season.csv
│   ├── batters_all.csv / bowlers_all.csv
│   ├── batters_development.csv / bowlers_development.csv
│   ├── batters_young.csv / bowlers_young.csv
│   ├── batters_pressure.csv / bowlers_pressure.csv
│   └── win_prob_curves.png
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## Official Top 10 Rankings & Old vs New Comparisons (Change 10)

### Set 1: Rajasthan Royals (RR Players Only)

#### Top 10 RR Batters (Min 30 Balls, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | % Below RRR | z(Innings Val) [Clipped] | z(% Below RRR) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Vaibhav Sooryavanshi** | India, India A, Rajasthan Royals | **1.2927** | 186 | 15 | 40.9% | +0.5012 | -2.0000 |
| **2** | **Yashasvi Jaiswal** | India, Rajasthan Royals | **1.2689** | 526 | 32 | 53.8% | +0.3412 | -2.0000 |
| **3** | **Jos Buttler** | England, Gujarat Titans, Rajasthan Royals | **1.2318** | 658 | 41 | 68.2% | +0.7234 | -1.1291 |
| **4** | **Devdutt Padikkal** | Lucknow Super Giants, Rajasthan Royals, Royal Challengers Bengaluru | **0.6752** | 321 | 17 | 69.2% | +0.2050 | -1.0666 |
| **5** | **Sanju Samson** | Chennai Super Kings, India, Rajasthan Royals | **0.6275** | 515 | 37 | 77.5% | +0.4127 | -0.5026 |
| **6** | **Riyan Parag** | India, Rajasthan Royals | **0.5915** | 367 | 27 | 73.6% | +0.2561 | -0.7675 |
| **7** | **Dhruv Jurel** | India, Rajasthan Royals | **0.4751** | 370 | 27 | 86.5% | +0.5679 | +0.1085 |
| **8** | **Rovman Powell** | Delhi Capitals, Kolkata Knight Riders, Rajasthan Royals, West Indies | **0.3983** | 700 | 53 | 83.1% | +0.3562 | -0.1182 |
| **9** | **Ravindra Jadeja** | Chennai Super Kings, India, Rajasthan Royals | **0.3733** | 357 | 24 | 89.6% | +0.5657 | +0.3221 |
| **10** | **Rassie van der Dussen** | Rajasthan Royals, South Africa | **0.3623** | 177 | 9 | 97.7% | +0.8595 | +0.8717 |

* **Old vs New Shift Audit:**
  * **Vaibhav Sooryavanshi:** Ranked #1 (Rating 1.2927, up 2 spots from #3). Exceptional 40.9% below RRR and strong innings value (+19.4).
  * **Yashasvi Jaiswal:** Ranked #2 (Rating 1.2689, up 7 spots from #9 — moved >5 places). Only 53.8% balls below RRR.
  * **Jos Buttler:** Ranked #3 (Rating 1.2318, up 3 spots from #6). 658 pressure balls across 41 innings.
  * **Devdutt Padikkal:** Ranked #4 (Rating 0.6752, NEW ENTRANT).
  * **Sanju Samson:** Ranked #5 (Rating 0.6275, up 2 spots from #7).
  * **Riyan Parag:** Ranked #6 (Rating 0.5915, NEW ENTRANT).
  * **Dhruv Jurel:** Ranked #7 (Rating 0.4751, down 2 spots from #5). High RRR lag (86.5%) applied a calibrated penalty.
  * **Rovman Powell:** Ranked #8 (Rating 0.3983, maintained rank).
  * **Ravindra Jadeja:** Ranked #9 (Rating 0.3733, down 5 spots from #4). High RRR lag (89.6%) reduced rating.
  * **Rassie van der Dussen:** Ranked #10 (Rating 0.3623, shifted down 8 spots from #2 — moved >5 places). RRR lag of 97.7% heavily penalized rating.
  * **Players who left Top 10:** Karun Nair (had only 5 innings < 8 innings floor), Dasun Shanaka (7 innings < 8 innings floor).

#### Top 10 RR Bowlers (Min 24 Band Balls, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | Band Balls | z(Runs Saved) [Clipped] | z(Wickets) [Clipped] | z(Last Over) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | Chennai Super Kings, Mumbai Indians, Rajasthan Royals | **3.0478** | 104 | 13 | 34 | +1.9191 | +2.0000 | -0.3357 |
| **2** | **FazalHaq Farooqi** | Afghanistan, Rajasthan Royals, Sunrisers Hyderabad | **1.2945** | 327 | 36 | 203 | +1.4113 | +0.5978 | -1.2392 |
| **3** | **Keshav Maharaj** | Mumbai Indians, Rajasthan Royals, South Africa | **1.1408** | 118 | 10 | 111 | +0.8556 | +0.5172 | +0.0000 |
| **4** | **Maheesh Theekshana** | Chennai Super Kings, Rajasthan Royals, Sri Lanka | **1.1322** | 483 | 43 | 424 | +0.9227 | -0.7097 | +1.9509 |
| **5** | **Jimmy Neesham** | New Zealand, Rajasthan Royals | **0.9654** | 88 | 15 | 32 | +0.2999 | +0.7103 | +0.4368 |
| **6** | **Trent Boult** | Mumbai Indians, New Zealand, Rajasthan Royals | **0.6338** | 570 | 48 | 463 | +0.9323 | -0.0644 | -0.4147 |
| **7** | **Yuzvendra Chahal** | India, Punjab Kings, Rajasthan Royals | **0.5318** | 535 | 45 | 481 | +0.4820 | -0.1833 | +0.5138 |
| **8** | **Ravi Bishnoi** | India, Lucknow Super Giants, Rajasthan Royals | **0.3586** | 712 | 55 | 670 | +0.7311 | -0.1790 | -0.3627 |
| **9** | **Kuldeep Sen** | Punjab Kings, Rajasthan Royals | **0.2970** | 131 | 10 | 96 | +1.2867 | -0.2618 | -1.3471 |
| **10** | **Ravichandran Ashwin** | Chennai Super Kings, India, Rajasthan Royals | **0.2701** | 414 | 26 | 414 | +1.0951 | -0.8094 | +0.0000 |

* **Old vs New Shift Audit:**
  * **Akash Madhwal:** Ranked #1 (Rating 3.0478, maintained rank).
  * **FazalHaq Farooqi:** Ranked #2 (Rating 1.2945, up 1 spot from #3).
  * **Keshav Maharaj:** Ranked #3 (Rating 1.1408, up 1 spot from #4).
  * **Maheesh Theekshana:** Ranked #4 (Rating 1.1322, up 1 spot from #5).
  * **Jimmy Neesham:** Ranked #5 (Rating 0.9654, up 1 spot from #6).
  * **Trent Boult:** Ranked #6 (Rating 0.6338, up 1 spot from #7).
  * **Yuzvendra Chahal:** Ranked #7 (Rating 0.5318, up 1 spot from #8).
  * **Ravi Bishnoi:** Ranked #8 (Rating 0.3586, up 2 spots from #10).
  * **Kuldeep Sen:** Ranked #9 (Rating 0.2970, maintained rank).
  * **Ravichandran Ashwin:** Ranked #10 (Rating 0.2701, NEW ENTRANT).
  * **Players who left Top 10:** Murugan Ashwin (had only 4 pressure innings < 8 innings floor).

---

### Set 2: Next Season Eligible (2025–2026 - Pakistan Excluded)

#### Top 10 Next Season Batters (Min 60 Balls / 30 RR, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | % Below RRR | z(Innings Val) [Clipped] | z(% Below RRR) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Phil Salt** | Delhi Capitals, England, Kolkata Knight Riders, Royal Challengers Bengaluru | **1.2630** | 331 | 21 | 60.1% | +0.5411 | -1.9017 |
| **2** | **Vaibhav Sooryavanshi** | India, India A, Rajasthan Royals | **1.2595** | 186 | 15 | 40.9% | +0.4627 | -2.0000 |
| **3** | **Yashasvi Jaiswal** | India, Rajasthan Royals | **1.2440** | 526 | 32 | 53.8% | +0.3150 | -2.0000 |
| **4** | **Jos Buttler** | England, Gujarat Titans, Rajasthan Royals | **1.2321** | 658 | 41 | 68.2% | +0.6678 | -1.2411 |
| **5** | **Virat Kohli** | India, Royal Challengers Bengaluru | **1.2058** | 602 | 30 | 69.8% | +0.7677 | -1.1165 |
| **6** | **Shubman Gill** | Gujarat Titans, India | **1.1698** | 656 | 35 | 70.4% | +0.7454 | -1.0628 |
| **7** | **Sai Sudharsan** | Gujarat Titans, India | **1.1249** | 560 | 25 | 74.6% | +0.8855 | -0.7197 |
| **8** | **Abhishek Sharma** | India, India A, Sunrisers Hyderabad | **1.1224** | 577 | 39 | 60.1% | +0.2890 | -1.9002 |
| **9** | **Ryan Rickelton** | Mumbai Indians, South Africa | **1.1187** | 338 | 23 | 56.2% | +0.3173 | -2.0000 |
| **10** | **Reeza Hendricks** | South Africa | **1.0908** | 396 | 16 | 81.6% | +1.1779 | -0.1562 |

* **Old vs New Shift Audit:**
  * **Phil Salt:** Ranked #1 (Rating 1.2630, NEW ENTRANT). High impact chase strike rate with 60.1% below RRR.
  * **Vaibhav Sooryavanshi:** Ranked #2 (Rating 1.2595, maintained rank).
  * **Yashasvi Jaiswal:** Ranked #3 (Rating 1.2440, NEW ENTRANT).
  * **Jos Buttler:** Ranked #4 (Rating 1.2321, NEW ENTRANT).
  * **Virat Kohli:** Ranked #5 (Rating 1.2058, NEW ENTRANT).
  * **Shubman Gill:** Ranked #6 (Rating 1.1698, NEW ENTRANT).
  * **Sai Sudharsan:** Ranked #7 (Rating 1.1249, up 2 spots from #9).
  * **Abhishek Sharma:** Ranked #8 (Rating 1.1224, NEW ENTRANT).
  * **Ryan Rickelton:** Ranked #9 (Rating 1.1187, NEW ENTRANT).
  * **Reeza Hendricks:** Ranked #10 (Rating 1.0908, shifted down 6 spots from #4 — moved >5 places).
  * **Players who left Top 10:** Atharwa Taide, Suryakumar Yadav, Tom Banton, Karun Nair (innings floor), Shashank Singh, Tilak Varma, KL Rahul.

#### Top 10 Next Season Bowlers (Min 48 Band Balls / 24 RR, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | Band Balls | z(Runs Saved) [Clipped] | z(Wickets) [Clipped] | z(Last Over) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | Chennai Super Kings, Mumbai Indians, Rajasthan Royals | **3.0686** | 104 | 13 | 34 | +1.8562 | +2.0000 | -0.1590 |
| **2** | **Mark Wood** | England, Lucknow Super Giants | **2.7452** | 123 | 10 | 88 | +1.4624 | +1.7322 | +1.2439 |
| **3** | **Mahedi Hasan** | Bangladesh | **2.6802** | 150 | 10 | 144 | +1.7278 | +1.8100 | +0.0000 |
| **4** | **Mitchell Santner** | Chennai Super Kings, Mumbai Indians, New Zealand | **2.3668** | 402 | 32 | 394 | +1.6764 | +0.9729 | +0.0000 |
| **5** | **Jacob Duffy** | New Zealand, Royal Challengers Bengaluru | **2.2683** | 290 | 27 | 208 | +0.9745 | +1.8851 | -0.4317 |
| **6** | **Gerald Coetzee** | Gujarat Titans, Mumbai Indians, South Africa, Sunrisers Hyderabad | **2.2557** | 136 | 17 | 71 | +0.4609 | +1.9658 | +1.2504 |
| **7** | **Axar Patel** | Delhi Capitals, India | **1.8833** | 580 | 46 | 520 | +1.0515 | +1.0647 | -0.1541 |
| **8** | **Nathan Ellis** | Australia, Chennai Super Kings, Punjab Kings | **1.8626** | 314 | 34 | 172 | +1.3255 | +0.5780 | +0.4877 |
| **9** | **Sikandar Raza** | Punjab Kings, Zimbabwe | **1.7980** | 232 | 18 | 220 | +1.7021 | -0.1590 | +1.2537 |
| **10** | **Krunal Pandya** | Lucknow Super Giants, Royal Challengers Bengaluru | **1.7957** | 373 | 25 | 343 | +1.1885 | -0.1618 | +2.0000 |

* **Old vs New Shift Audit:**
  * **Akash Madhwal (#1), Mark Wood (#2), Mahedi Hasan (#3), Mitchell Santner (#4), Jacob Duffy (#5)** all maintained top ranks.
  * **Gerald Coetzee (#6), Sikandar Raza (#9), Krunal Pandya (#10)** are NEW ENTRANTS.
  * **Axar Patel:** Ranked #7 (up 3 spots).
  * **Nathan Ellis:** Ranked #8 (maintained rank).
  * **Players who left Top 10:** Mohammad Rishad Hossain, Corbin Bosch, FazalHaq Farooqi.

---

### Set 3: All Senior Players (Global Senior Pool - Pakistan Included)

#### Top 10 All Senior Batters (Min 60 Balls / 30 RR, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | % Below RRR | z(Innings Val) [Clipped] | z(% Below RRR) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Phil Salt** | Delhi Capitals, England, Kolkata Knight Riders, Royal Challengers Bengaluru | **1.2881** | 331 | 21 | 60.1% | +0.5368 | -1.9696 |
| **2** | **Vaibhav Sooryavanshi** | India, India A, Rajasthan Royals | **1.2563** | 186 | 15 | 40.9% | +0.4590 | -2.0000 |
| **3** | **Jos Buttler** | England, Gujarat Titans, Rajasthan Royals | **1.2526** | 658 | 41 | 68.2% | +0.6624 | -1.2947 |
| **4** | **Yashasvi Jaiswal** | India, Rajasthan Royals | **1.2416** | 526 | 32 | 53.8% | +0.3124 | -2.0000 |
| **5** | **Virat Kohli** | India, Royal Challengers Bengaluru | **1.2233** | 602 | 30 | 69.8% | +0.7616 | -1.1674 |
| **6** | **Mohammad Rizwan** | Pakistan | **1.2008** | 413 | 16 | 83.8% | +1.3741 | -0.0023 |
| **7** | **Shubman Gill** | Gujarat Titans, India | **1.1871** | 656 | 35 | 70.4% | +0.7394 | -1.1125 |
| **8** | **Abhishek Sharma** | India, India A, Sunrisers Hyderabad | **1.1510** | 577 | 39 | 60.1% | +0.2866 | -1.9681 |
| **9** | **Sai Sudharsan** | Gujarat Titans, India | **1.1375** | 560 | 25 | 74.6% | +0.8784 | -0.7619 |
| **10** | **Ryan Rickelton** | Mumbai Indians, South Africa | **1.1166** | 338 | 23 | 56.2% | +0.3148 | -2.0000 |

* **Top Performers:** Phil Salt (#1, 1.2881), Vaibhav Sooryavanshi (#2, 1.2563), Jos Buttler (#3, 1.2526), Yashasvi Jaiswal (#4, 1.2416), Virat Kohli (#5, 1.2233), Mohammad Rizwan (#6, 1.2008), Shubman Gill (#7, 1.1871), Abhishek Sharma (#8, 1.1510), Sai Sudharsan (#9, 1.1375), Ryan Rickelton (#10, 1.1166).

#### Top 10 All Senior Bowlers (Min 48 Band Balls / 24 RR, Min 8 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | Band Balls | z(Runs Saved) [Clipped] | z(Wickets) [Clipped] | z(Last Over) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | Chennai Super Kings, Mumbai Indians, Rajasthan Royals | **3.0363** | 104 | 13 | 34 | +1.8469 | +2.0000 | -0.2199 |
| **2** | **Mark Wood** | England, Lucknow Super Giants | **2.6583** | 123 | 10 | 88 | +1.4476 | +1.6970 | +1.1024 |
| **3** | **Mahedi Hasan** | Bangladesh | **2.6443** | 150 | 10 | 144 | +1.7167 | +1.7738 | +0.0000 |
| **4** | **Mohammad Nawaz** | Pakistan | **2.3752** | 250 | 18 | 246 | +0.8313 | +2.0000 | +0.0000 |
| **5** | **Mitchell Santner** | Chennai Super Kings, Mumbai Indians, New Zealand | **2.3336** | 402 | 32 | 394 | +1.6646 | +0.9476 | +0.0000 |
| **6** | **Jacob Duffy** | New Zealand, Royal Challengers Bengaluru | **2.1984** | 290 | 27 | 208 | +0.9528 | +1.8479 | -0.4770 |
| **7** | **Gerald Coetzee** | Gujarat Titans, Mumbai Indians, South Africa, Sunrisers Hyderabad | **2.1536** | 136 | 17 | 71 | +0.4320 | +1.9275 | +1.1085 |
| **8** | **Axar Patel** | Delhi Capitals, India | **1.8116** | 580 | 46 | 520 | +1.0309 | +1.0382 | -0.2153 |
| **9** | **Nathan Ellis** | Australia, Chennai Super Kings, Punjab Kings | **1.7881** | 314 | 34 | 172 | +1.3087 | +0.5579 | +0.3897 |
| **10** | **Krunal Pandya** | Lucknow Super Giants, Royal Challengers Bengaluru | **1.7486** | 373 | 25 | 343 | +1.1698 | -0.1722 | +1.9521 |

* **Top Performers:** Akash Madhwal (#1, 3.0363), Mark Wood (#2, 2.6583), Mahedi Hasan (#3, 2.6443), Mohammad Nawaz (#4, 2.3752), Mitchell Santner (#5, 2.3336), Jacob Duffy (#6, 2.1984), Gerald Coetzee (#7, 2.1536), Axar Patel (#8, 1.8116), Nathan Ellis (#9, 1.7881), Krunal Pandya (#10, 1.7486).

---

### Development Tier (Mens T20 Emerging Teams Asia Cup 2024 - Comp 2670)

#### Top Development Batters (Min 30 Balls, Min 2 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | % Below RRR | z(Innings Val) [Clipped] | z(% Below RRR) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Sediqullah Atal** | Afghanistan A | **1.4471** | 55 | 2 | 81.8% | +1.7699 | -0.9331 |
| **2** | **Babar Hayat** | Hong Kong | **0.8079** | 52 | 2 | 80.8% | +0.7475 | -1.0531 |
| **3** | **Ramandeep Singh** | India A | **0.1322** | 37 | 2 | 97.3% | +0.6586 | +0.8384 |
| **4** | **Yasim Murtaza** | Hong Kong | **-0.2266** | 36 | 2 | 100.0% | +0.1585 | +1.1478 |

* **Top Performers:** Sediqullah Atal (#1, 1.4471), Babar Hayat (#2, 0.8079), Ramandeep Singh (#3, 0.1322), Yasim Murtaza (#4, -0.2266).

#### Top Development Bowlers (Min 24 Band Balls, Min 2 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | Band Balls | z(Runs Saved) [Clipped] | z(Wickets) [Clipped] | z(Last Over) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Eshan Malinga** | Sri Lanka A | **1.5511** | 42 | 5 | 24 | +0.7504 | +1.6871 | +0.0000 |
| **2** | **Shahnawaz Dhani** | Pakistan A | **1.0753** | 36 | 2 | 36 | +1.8222 | -0.0301 | +0.0000 |
| **3** | **Allah Ghazanfar** | Afghanistan A | **0.4138** | 39 | 3 | 36 | +0.5171 | +0.1514 | +0.0000 |
| **4** | **Faridoon Dawoodzai** | Afghanistan A | **0.1649** | 32 | 2 | 26 | -1.0451 | +1.3337 | +0.0000 |
| **5** | **Nimesh Vimukthi** | Sri Lanka A | **0.0492** | 42 | 3 | 42 | +0.1018 | -0.0246 | +0.0000 |
| **6** | **Sharafuddin Ashraf** | Afghanistan A | **-0.4260** | 33 | 2 | 33 | +0.0821 | -0.8178 | +0.0000 |
| **7** | **Karim Janat** | Afghanistan A | **-1.0373** | 34 | 4 | 30 | -0.7564 | -1.0131 | +0.0000 |
| **8** | **Bilal Sami** | Afghanistan A | **-1.6366** | 35 | 3 | 24 | -1.4722 | -1.2866 | +0.0000 |

* **Top Performers:** Eshan Malinga (#1, 1.5511), Shahnawaz Dhani (#2, 1.0753), Allah Ghazanfar (#3, 0.4138), Faridoon Dawoodzai (#4, 0.1649), Nimesh Vimukthi (#5, 0.0492), Sharafuddin Ashraf (#6, -0.4260), Karim Janat (#7, -1.0373), Bilal Sami (#8, -1.6366).

---

### Youth Tier (Asia Cup Rising Stars 2025 - Comp 3289)

#### Top Youth Batters (Min 30 Balls, Min 2 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | % Below RRR | z(Innings Val) [Clipped] | z(% Below RRR) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Nehal Wadhera** | India A | **0.3857** | 42 | 2 | 88.1% | +0.6612 | +0.0000 |

* **Top Performer:** Nehal Wadhera (#1, 0.3857, 42 balls across 2 innings). Sohaib Khan (43 balls, 1 innings) highlighted with single-innings note.

#### Top Youth Bowlers (Min 24 Band Balls, Min 2 Innings)
| Rank | Player | Team(s) | Adjusted Rating | Pressure Balls | Pressure Inns | Band Balls | z(Runs Saved) [Clipped] | z(Wickets) [Clipped] | z(Last Over) [Clipped] |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Garuka Sanketh** | Sri Lanka A | **0.0000** | 52 | 4 | 29 | +0.0000 | +0.0000 | +0.0000 |

* **Top Performer:** Garuka Sanketh (#1, 0.0000, 52 pressure balls, 4 innings).

---

## 7. Sanity Checks & Verification (Change 8)

The pipeline automates verification assertions upon every execution:
1. **Strict Tier Separation:** Confirmed 0 match ID overlap between Senior matches (757) and Development/Youth matches (30).
2. **Synthetic Sanity Check (a) - Batter Dismissals:** Batter with same runs and more dismissals ranks strictly lower ($0.0 < 20.0$). `[PASSED]`
3. **Synthetic Sanity Check (b) - Bowler Conceding Runs:** Bowler conceding more runs in band ranks strictly lower ($-1.0 < +2.0$). `[PASSED]`
4. **Synthetic Sanity Check (c) - RRR Strike Rate Lag:** Batter lagging RRR in $>50\%$ balls ranks strictly lower ($0.333 < 2.000$). `[PASSED]`
5. **Threshold Compliance:** Confirmed 100% of players in all 10 Top-10 lists satisfy both their minimum delivery threshold and innings floor. `[PASSED]`
6. **Data Privacy & Integrity:** Zero raw competition files are committed to version control.

# SupeRR Selector: Rajasthan Royals Pressure Analysis & Dual Ranking Engine

A comprehensive decision science and talent identification system developed for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure using state-dependent expected-value baselines constructed from all qualifying IPL and T20 International matches (2022 onward).

The repository features two independent, internally consistent ranking sets:
* **Set A (All Players):** Global ranking across all qualifying players in the 2022+ competition dataset.
* **Set B (RR Players Only):** Franchise-specific ranking filtered to players who have appeared for Rajasthan Royals (`team_id == 3`).

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Standalone interactive dashboard (Chart.js, embedded arrays, RR palette)
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory analysis
│   ├── win_probability_model.py   # Second-innings logistic regression model & swing thresholding
│   └── pressure_metrics.py        # Expected-value baseline engine, dual z-scoring, Bayesian shrinkage & export
│
├── data/
│   ├── expected_tables.csv        # 413 empirical game-state expected value buckets
│   ├── top10_all_batters.csv      # Top 10 Batters (Global Pool)
│   ├── top10_all_bowlers.csv      # Top 10 Bowlers (Global Pool)
│   ├── top10_rr_batters.csv       # Top 10 Batters (Rajasthan Royals)
│   ├── top10_rr_bowlers.csv       # Top 10 Bowlers (Rajasthan Royals)
│   ├── batters_all.csv            # Full evaluation metrics for 888 batters globally
│   ├── bowlers_all.csv            # Full evaluation metrics for 646 bowlers globally
│   ├── batters_rr.csv             # Full evaluation metrics for 57 RR batters
│   ├── bowlers_rr.csv             # Full evaluation metrics for 45 RR bowlers
│   ├── batters_pressure.csv       # Backward-compatible RR batters table
│   ├── bowlers_pressure.csv       # Backward-compatible RR bowlers table
│   └── win_prob_curves.png        # In-play win probability validation curves with high-pressure deliveries
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## 1. Player Count Reconciliation

A rigorous reconciliation across all player definitions was conducted to audit contract mappings against on-field matchday participation and statistical eligibility:

| Category | Set A: All Players | Set B: Rajasthan Royals Only | Definition & Methodology |
| :--- | :---: | :---: | :--- |
| **Squad Roster** | **1,217** | **69** | Total unique player IDs registered in `competition_team_player_mapping.csv` across competitions (2022 onward). Rajasthan Royals verified as `team_id == 3`. |
| **Batted in Matches** | **888** | **57** | Unique players who faced $\ge 1$ delivery as striker in ball-by-ball match records. |
| **Bowled in Matches** | **647** | **45** | Unique players who bowled $\ge 1$ delivery in ball-by-ball match records. |
| **Batted or Bowled** | **958** | **60** | Unique players with active playing participation on the pitch. |
| **Non-Playing / Reserves** | **259** | **9** | Contracted squad members who recorded zero deliveries faced or bowled. |
| **Qualifying Batters** | **443** | **33** | Batters meeting the sample threshold of $\ge 30$ pressure deliveries faced. |
| **Qualifying Bowlers** | **285** | **34** | Bowlers meeting the sample threshold of $\ge 24$ deliveries in the RRR 8–12 pressure band. |

---

## 2. Expected-Value Baselines (Task 1)

Rather than evaluating players in a vacuum or comparing them only against uncontrolled averages, performance is evaluated against state-dependent expected values computed across **all 198,032 deliveries** in 838 valid IPL and T20I matches (2022 onward).

### Second Innings (Chase) State Partitioning
Each delivery in the second innings is mapped to a discrete game-state bucket based on three orthogonal dimensions:
1. **Balls Remaining:** 6 discrete phases:
   * `1–6`: Final over
   * `7–12`: Penultimate over (Over 19)
   * `13–24`: Early death overs (Overs 17–18)
   * `25–36`: Late middle overs (Overs 14–15)
   * `37–60`: Mid-innings phase (Overs 10–13)
   * `61–120`: Powerplay and early accumulation (Overs 1–10)
2. **Required Run Rate (RRR):** 0.5 runs-per-over intervals capped at 16+:
   * `<6.0`, `6.0–6.5`, `6.5–7.0`, `7.0–7.5`, `7.5–8.0`, `8.0–8.5`, `8.5–9.0`, `9.0–9.5`, `9.5–10.0`, `10.0–10.5`, `10.5–11.0`, `11.0–11.5`, `11.5–12.0`, `12.0–12.5`, `12.5–13.0`, `13.0–13.5`, `13.5–14.0`, `14.0–14.5`, `14.5–15.0`, `15.0–15.5`, `15.5–16.0`, `16.0+`.
3. **Wickets in Hand:** 3 leverage tiers:
   * `1–3`: Tail exposed / collapse state
   * `4–6`: Middle order / moderate cushion
   * `7–10`: Deep batting resources / low risk

### First Innings State Partitioning
Because the first innings has no fixed chase target or real-time required run rate, it is bucketed by:
1. **Balls Remaining & Over Stage:** Powerplay (overs 1–6), Middle overs (overs 7–15), Death overs (overs 16–20).
2. **Venue Par:** Average first-innings score by venue and season, classified into tertiles:
   * `Low (<165)`: Bowling-friendly or testing surfaces (260 matches)
   * `Medium (165–185)`: Balanced, standard T20 surfaces (304 matches)
   * `High (>185)`: High-scoring batting paradises (274 matches)

For all 413 discrete buckets (saved in `data/expected_tables.csv`), the baseline tracks:
* **Batter Expectation:** `batter_avg_runs_per_ball`, `batter_dot_pct`, `batter_boundary_pct`, and `batter_dismissal_rate`.
* **Bowler Expectation:** `bowler_avg_runs_per_ball`, `bowler_dot_pct`, `bowler_wicket_rate`, and `runs_per_wicket`.

---

## 3. Pressure Definitions (Task 2)

Match pressure is formally defined at the delivery level:

* **Batting Pressure:**
  * Second-innings deliveries where **Required Run Rate $\ge 9.0$**, OR
  * **Death overs (overs 16–20 / balls remaining $\le 30$)** in either innings.
  *(Captures 92,767 high-leverage deliveries across the dataset).*
* **Bowling Pressure:**
  * **Core Chase Band:** Second-innings deliveries where RRR is between **8.0 and 12.0** with balls remaining $> 6$ (41,652 deliveries).
  * **Last-Over Component:** Balls remaining **1–6** of either innings (14,358 deliveries).
  * **Exclusivity Rule:** A ball qualifying for both counts **only once**, categorized under the last-over component.
* **Win-Probability Swing Benchmark:**
  * Delivery-level win probability swing $|P_{\text{after}} - P_{\text{before}}|$ is evaluated using the logistic regression model trained on 2022–2024 matches. Deliveries in the top 15% ($\ge 85\text{th percentile} \approx 0.02539$ swing) are maintained as an empirical robustness benchmark.

---

## 4. Rating Formulas & Component Mechanics

### Batter Rating Formula (Task 3)
For each batter's pressure deliveries:
1. **Runs Above Expected per 100 Balls:**
   $$\text{runs\_above\_expected\_per\_100} = \frac{\sum (\text{runs}_i - E[\text{runs}]_i)}{N_{\text{press}}} \times 100$$
2. **Excess Dismissals & Dismissal Penalty:**
   $$\Delta D = D_{\text{actual}} - \sum E[\text{dismissal}]_i$$
   $$\text{dismissal\_penalty} = \left(\frac{\Delta D}{\text{Innings}_{\text{press}}}\right) \times \left(\frac{\sum E[\text{runs}]_i}{\text{Innings}_{\text{press}}}\right)$$
   *(Getting out more frequently than expected imposes a heavy penalty equal to the expected runs that lost wicket would have yielded).*
3. **Standardized Raw Rating:**
   $$\text{Raw Rating}_{\text{batter}} = z(\text{runs\_above\_expected\_per\_100}) - z(\text{dismissal\_penalty})$$
4. **Bayesian Shrinkage Toward Zero ($k = 30$):**
   $$\text{Adjusted Rating}_{\text{batter}} = \frac{N_{\text{press}} \cdot \text{Raw Rating}}{N_{\text{press}} + 30}$$
   *Eligibility:* Minimum 30 pressure deliveries.

### Bowler Rating Formula (Task 4)
For each bowler's pressure deliveries:
1. **Core Band (RRR 8–12, Balls Rem $> 6$):**
   * Runs Saved per Over:
     $$\text{runs\_saved\_per\_over} = \frac{\sum (E[\text{runs}]_i - \text{actual\_runs}_i)}{N_{\text{band}} / 6.0}$$
   * Wicket Component (weighted by bucket runs per wicket):
     $$\text{wicket\_component} = \frac{\sum (\text{wicket}_i - E[\text{wicket}]_i) \times W_{B_i}}{N_{\text{band}} / 6.0}$$
2. **Last-Over Component (Balls Rem $1–6$):**
   * Computes runs saved per over and weighted wicket component on last-over buckets:
     $$\text{last\_over\_score} = \text{lo\_runs\_saved} + \text{lo\_wicket\_component}$$
   * Standardized as $z(\text{last\_over\_component})$ for bowlers with $\ge 12$ last-over balls (assigned $0.0$ for limited sample).
3. **Standardized Raw Rating:**
   $$\text{Raw Rating}_{\text{bowler}} = z(\text{runs\_saved\_per\_over}) + z(\text{wicket\_component}) + 0.5 \cdot z(\text{last\_over\_component})$$
4. **Bayesian Shrinkage Toward Zero ($k = 24$):**
   $$\text{Adjusted Rating}_{\text{bowler}} = \frac{N_{\text{press}} \cdot \text{Raw Rating}}{N_{\text{press}} + 24}$$
   *Eligibility:* Minimum 24 balls in the 8–12 band (and $\ge 12$ last-over balls for last-over score).

---

## 5. Z-Score Pools Choice (Task 5)

Z-scores are calculated independently within each ranking set:
* **Set A (All Players):** Scaled using the **global qualifying player population** mean and standard deviation. This provides an absolute worldwide benchmark across international and IPL talent.
* **Set B (RR Players Only):** Scaled using the **Rajasthan Royals qualifying squad** mean and standard deviation. This provides franchise-specific rankings where ratings reflect each player's standing relative to Rajasthan Royals squad peers.

Both sets share the exact same underlying expected-value baseline table (`expected_tables.csv`).

---

## 6. Official Top 10 Rankings

### Top 10 Batters (Set A: All Players)
*Minimum 30 pressure balls*

| Rank | Player | Team(s) | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Usman Ghani** | Afghanistan | **5.2167** | -49.99 | -69.20 | 59 |
| **2** | **Arafat Minhas** | Pakistan, Pakistan A | **4.0609** | +11.47 | -35.90 | 82 |
| **3** | **Aamir Kaleem** | Oman | **3.3952** | -31.02 | -42.74 | 67 |
| **4** | **Rizwan CP** | United Arab Emirates | **3.3936** | -2.47 | -39.41 | 48 |
| **5** | **Atharwa Taide** | Punjab Kings, Sunrisers Hyderabad | **3.3193** | -17.52 | -34.06 | 111 |
| **6** | **Maaz Sadaqat** | Pakistan, Pakistan A | **3.2700** | +43.49 | -29.51 | 38 |
| **7** | **Vaibhav Sooryavanshi** | India, India A, Rajasthan Royals | **3.0722** | +93.48 | +0.84 | 210 |
| **8** | **Rony Talukder** | Bangladesh | **2.5642** | -6.94 | -36.13 | 33 |
| **9** | **Shan Masood** | Pakistan | **2.3971** | -14.50 | -24.46 | 119 |
| **10** | **Suryakumar Yadav** | India, Mumbai Indians | **2.3144** | +32.76 | -7.23 | 1177 |

---

### Top 10 Batters (Set B: Rajasthan Royals Only)
*Minimum 30 pressure balls*

| Rank | Player | Team(s) | Adjusted Rating | Runs Above /100b | Dismissal Penalty | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Karun Nair** | Delhi Capitals, Lucknow Super Giants, Rajasthan Royals | **2.9980** | +31.70 | -8.70 | 89 |
| **2** | **Vaibhav Sooryavanshi** | India, India A, Rajasthan Royals | **2.1779** | +93.48 | +0.84 | 210 |
| **3** | **Rassie van der Dussen** | Rajasthan Royals, South Africa | **2.0826** | -6.67 | -7.26 | 217 |
| **4** | **Ravindra Jadeja** | Chennai Super Kings, India, Rajasthan Royals | **1.7956** | -12.09 | -6.29 | 720 |
| **5** | **Dhruv Jurel** | India, Rajasthan Royals | **1.7780** | +3.04 | -5.05 | 510 |
| **6** | **Jos Buttler** | England, Gujarat Titans, Rajasthan Royals | **1.3316** | +25.97 | -1.67 | 922 |
| **7** | **Sanju Samson** | Chennai Super Kings, India, Rajasthan Royals | **1.0282** | +20.03 | -1.30 | 741 |
| **8** | **Rovman Powell** | DC, KKR, Rajasthan Royals, West Indies | **0.9802** | +12.96 | -1.72 | 1049 |
| **9** | **Yashasvi Jaiswal** | India, Rajasthan Royals | **0.8976** | +32.51 | +0.11 | 579 |
| **10** | **Dasun Shanaka** | Gujarat Titans, Rajasthan Royals, Sri Lanka | **0.7975** | +4.54 | -1.95 | 560 |

---

### Top 10 Bowlers (Set A: All Players)
*Minimum 24 balls in RRR 8–12 band*

| Rank | Player | Team(s) | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | CSK, Mumbai Indians, Rajasthan Royals | **4.7521** | +1.90 | +19.50 | -0.78 | 104 |
| **2** | **Qais Ahmad** | Afghanistan, Afghanistan A | **3.7165** | +2.94 | +17.08 | 0.00 | 37 |
| **3** | **Mark Wood** | England, Lucknow Super Giants | **2.5838** | +1.44 | +6.17 | +8.21 | 123 |
| **4** | **Charith Asalanka** | Mumbai Indians, Sri Lanka | **2.5149** | +1.88 | +14.10 | 0.00 | 28 |
| **5** | **Kartik Meiyappan** | United Arab Emirates | **2.3633** | +3.65 | +8.51 | 0.00 | 25 |
| **6** | **Mohammad Rishad Hossain** | Bangladesh | **2.3408** | +0.51 | +1.24 | +25.55 | 224 |
| **7** | **Dunith Wellalage** | Sri Lanka, Sri Lanka A | **2.2153** | +1.14 | +6.35 | +4.75 | 122 |
| **8** | **Ebadot Hossain** | Bangladesh | **2.1788** | +1.36 | +8.36 | +4.04 | 50 |
| **9** | **Lalit Yadav** | Delhi Capitals | **2.1307** | +2.89 | +8.89 | 0.00 | 25 |
| **10** | **Mitchell Santner** | CSK, Mumbai Indians, New Zealand | **1.9745** | +1.84 | +3.14 | +6.25 | 453 |

---

### Top 10 Bowlers (Set B: Rajasthan Royals Only)
*Minimum 24 balls in RRR 8–12 band*

| Rank | Player | Team(s) | Adjusted Rating | Runs Saved /Over | Wicket Comp | Last-Over Score | Pressure Balls |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Akash Madhwal** | CSK, Mumbai Indians, Rajasthan Royals | **4.8247** | +1.90 | +19.50 | -0.78 | 104 |
| **2** | **FazalHaq Farooqi** | Afghanistan, Rajasthan Royals, SRH | **1.8657** | +1.34 | +4.52 | -4.42 | 361 |
| **3** | **Murugan Ashwin** | Mumbai Indians, Rajasthan Royals | **1.6279** | +1.16 | +4.20 | 0.00 | 83 |
| **4** | **Corbin Bosch** | Mumbai Indians, Rajasthan Royals, SA | **1.5925** | +0.30 | +7.96 | -1.96 | 118 |
| **5** | **Keshav Maharaj** | Mumbai Indians, Rajasthan Royals, SA | **1.1992** | +0.71 | +3.11 | -20.57 | 118 |
| **6** | **Maheesh Theekshana** | CSK, Rajasthan Royals, Sri Lanka | **1.1770** | +0.85 | -2.35 | +9.46 | 489 |
| **7** | **Trent Boult** | Mumbai Indians, New Zealand, RR | **0.9188** | +0.86 | +0.59 | -1.47 | 570 |
| **8** | **Yuzvendra Chahal** | India, Punjab Kings, Rajasthan Royals | **0.8155** | +0.48 | -0.15 | +3.52 | 543 |
| **9** | **Kuldeep Sen** | Punjab Kings, Rajasthan Royals | **0.6442** | +1.23 | -0.25 | -6.01 | 131 |
| **10** | **Ravi Bishnoi** | India, LSG, Rajasthan Royals | **0.5785** | +0.63 | -0.03 | -1.61 | 719 |

---

## 7. Sanity Checks & Verification (Task 6)

The rating engine was subjected to strict sanity checks:
1. **Batter Dismissal Penalty Sign Check:**
   $$\frac{\partial (\text{Raw Rating})}{\partial (\text{Dismissal Penalty})} = -\frac{1}{\sigma_{\text{penalty}}} < 0$$
   Holding runs scored equal, any increase in dismissals increases the dismissal penalty and strictly lowers the rating (verified correlation $r = -1.000$).
2. **Bowler Runs Conceded Sign Check:**
   $$\frac{\partial (\text{Raw Rating})}{\partial (\text{Runs Conceded})} = -\frac{1}{\text{Overs} \cdot \sigma_{\text{runs\_saved}}} < 0$$
   Holding wickets taken equal, any increase in runs conceded reduces runs saved per over and strictly lowers the rating (verified correlation $r = +1.000$).
3. **Tail Analysis:**
   The bottom 5 qualifying batters in the pool (e.g. Varun Chakravarthy, Nathan Ellis, Maheesh Theekshana) have negative runs above expected and elevated dismissal rates under pressure, confirming that tailenders facing 30+ pressure balls appropriately sit at the bottom.

---

## 8. Dashboard Features (Task 7)

The interactive dashboard (`frontend/index.html` and root `index.html`) is self-contained with zero external fetches:
* **Pool Switcher:** One-click toggle between "RR Players Only" and "All Players (Global)" updating all leaderboards, dropdowns, and tables.
* **Component Cards:** Explicitly surfaces Runs Above Expected, Dismissal Penalty, Runs Saved Per Over, Wicket Component, and Last-Over Component.
* **RR Visual Palette:** High-contrast Rajasthan Royals Pink (`#E91E8C`), Deep Blue (`#1A3C8F`), Light Blue (`#EAF2FB`), and Dark Navy (`#14213D`).
* **Sortable Roster Table:** Instant search and sorting across all batters and bowlers with reliable sample indicators.

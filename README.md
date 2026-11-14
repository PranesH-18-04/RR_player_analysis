# SupeRR Selector: Rajasthan Royals Pressure Analysis & Win Probability Model

A data-driven talent identification and decision science system built for the **Rajasthan Royals - SupeRR Selector (Season 2)** hackathon. This system quantifies player performance under acute match pressure compared with baseline/normal play across IPL and T20 International matches (2022 onward).

---

## Repository Structure

```
RR_player_analysis/
│
├── frontend/
│   └── index.html                 # Interactive RR Pressure Analytics Dashboard (Chart.js, RR jersey theme)
│
├── backend/
│   ├── exploratory_analysis.py    # Player roster reconciliation & season exploratory data analysis
│   ├── win_probability_model.py   # Second-innings win probability model (Logistic Regression & swing labels)
│   └── pressure_metrics.py        # Pressure metrics, dismissal penalties, z-score rating & shrinkage
│
├── data/
│   ├── batters_pressure.csv       # Pressure metrics & adjusted ratings for 57 RR batters
│   ├── bowlers_pressure.csv       # Pressure metrics & adjusted ratings for 45 RR bowlers
│   ├── top10_rankings.csv         # Top 10 Batters and Top 10 Bowlers by Adjusted Rating
│   └── win_prob_curves.png        # Second-innings in-play win probability validation curves
│
├── index.html                     # Root entry point for GitHub Pages deployment
├── .gitignore
└── README.md
```

---

## 1. Player Count Reconciliation

A rigorous reconciliation across all player definitions was conducted to distinguish official squad contracts from active on-field participation and statistically reliable samples:

| Definition | Player Count | Explanation |
| :--- | :---: | :--- |
| **Squad Roster** | **69** | Total unique players mapped to Rajasthan Royals in `competition_team_player_mapping.csv` across the 2022–2026 IPL editions. |
| **Batted in Matches** | **57** | Unique RR players who faced $\ge 1$ delivery as a striker in competitive match records. |
| **Bowled in Matches** | **45** | Unique RR players who bowled $\ge 1$ delivery in competitive match records. |
| **Both Batted & Bowled** | **42** | Players who contributed with both bat and ball (all-rounders, bowling all-rounders, and tailenders). |
| **Reserves / Non-Playing** | **9** | Contracted squad members who never took the pitch to face or bowl a ball in matchday records. |
| **Qualifying Batters** | **28** | RR batters meeting the minimum sample-size rule ($\ge 30$ pressure deliveries faced). |
| **Qualifying Bowlers** | **32** | RR bowlers meeting the minimum sample-size rule ($\ge 24$ pressure deliveries bowled). |

---

## 2. Win-Probability-Based Pressure Definition

Rather than relying purely on static heuristics, match pressure is grounded in empirical game state leverage:

1. **Second Innings (Chasing Side):**
   * A Logistic Regression win probability model is trained on historical 2022–2024 matches using four core state features:
     * `runs_needed`: Target (including D/L revised targets) minus cumulative runs.
     * `balls_remaining`: Balls allowed (adjusted for D/L revised overs) minus deliveries bowled.
     * `wickets_in_hand`: $10 - \text{wickets lost}$.
     * `rrr`: Required run rate $(\text{runs\_needed} \times 6 / \text{balls\_remaining})$.
   * For every delivery, the win probability swing is computed:
     $$\Delta P = |P_{\text{after}} - P_{\text{before}}|$$
   * A delivery is classified as **High Pressure** if its swing falls within the **top 15%** ($\ge 85\text{th percentile}$, threshold $\approx 0.02539$ or $2.54\%$ swing) of all training set deliveries.
2. **First Innings:**
   * Uses the **death-overs rule (overs 16–20)**. In the first innings, there is no fixed chasing target or required run rate to anchor real-time win probability; overs 16–20 represent the designated resource-maximization phase where boundary acceleration and wicket loss have the highest mathematical impact on par score expectations.
3. **Documented Comparison:**
   * Rule-based pressure (death overs or chase RRR $\ge 9.0$) is tracked in parallel for audit and baseline benchmarking.

---

## 3. Performance Metrics & Combined Rating Formula

### Batter Metrics & Dismissal Penalties
* Evaluates `pressure_strike_rate` vs `normal_strike_rate` ($\text{sr\_difference} = \text{SR}_{\text{pressure}} - \text{SR}_{\text{normal}}$).
* Evaluates dismissal frequency:
  * `pressure_dismissal_rate` = dismissals per pressure innings.
  * `normal_dismissal_rate` = dismissals per normal innings.
  * $\text{dismissal\_difference} = \text{Rate}_{\text{pressure}} - \text{Rate}_{\text{normal}}$.
* **Batter Rating:**
  $$\text{Rating}_{\text{batter}} = z(\text{sr\_difference}) - z(\text{dismissal\_difference})$$
  *(Getting dismissed more frequently under pressure penalizes the rating).*

### Bowler Metrics & Wicket Rates
* Evaluates `pressure_economy` vs `normal_economy` ($\text{economy\_difference} = \text{Econ}_{\text{pressure}} - \text{Econ}_{\text{normal}}$).
* Evaluates wicket-taking potency:
  * `pressure_wickets_per_over` vs `normal_wickets_per_over`.
  * $\text{wicket\_difference} = \text{WPO}_{\text{pressure}} - \text{WPO}_{\text{normal}}$.
* **Bowler Rating:**
  $$\text{Rating}_{\text{bowler}} = z(\text{wicket\_difference}) - z(\text{economy\_difference})$$
  *(Conceding more runs under pressure penalizes the rating, while higher pressure wicket-taking elevates it).*

### Empirical Bayesian Shrinkage
To prevent small-sample anomalies from topping the leaderboards, ratings are shrunk toward the squad mean:
$$\text{Adjusted Rating} = \frac{\text{pressure\_balls} \cdot \text{Rating} + k \cdot \mu_{\text{squad}}}{\text{pressure\_balls} + k}$$
* $k = 30$ for batters.
* $k = 24$ for bowlers.

---

## 4. Top 10 Rankings (Qualifying Players)

The official rankings produced by `backend/pressure_metrics.py` and stored in `data/top10_rankings.csv`:

### Top 10 Batters (Minimum 30 Pressure Balls)
| Rank | Player Name | Pressure Balls | SR Diff | Dismissal Diff | Raw Rating | Adjusted Rating |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Yashasvi Jaiswal** | 209 | +235.0 | +0.348 | +2.0437 | **+1.7872** |
| **2** | **Vaibhav Sooryavanshi** | 99 | +243.6 | +0.330 | +2.3032 | **+1.7676** |
| **3** | **Dhruv Jurel** | 201 | +128.4 | +0.155 | +1.7885 | **+1.5563** |
| **4** | **Ravichandran Ashwin** | 94 | +62.4 | -0.022 | +2.0399 | **+1.5464** |
| **5** | **Jofra Archer** | 79 | +50.9 | +0.139 | +0.7181 | **+0.5205** |
| **6** | **Trent Boult** | 54 | +36.3 | +0.107 | +0.7228 | **+0.4647** |
| **7** | **Devdutt Padikkal** | 110 | +164.0 | +0.410 | +0.5174 | **+0.4065** |
| **8** | **Rassie van der Dussen** | 70 | +152.7 | +0.381 | +0.5512 | **+0.3858** |
| **9** | **Karun Nair** | 35 | +245.0 | +0.556 | +0.7159 | **+0.3855** |
| **10** | **Sanju Samson** | 314 | +101.1 | +0.304 | +0.3107 | **+0.2836** |

### Top 10 Bowlers (Minimum 24 Pressure Balls)
| Rank | Player Name | Pressure Balls | Wicket Diff | Econ Diff | Raw Rating | Adjusted Rating |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Jimmy Neesham** | 117 | +0.971 | +3.39 | +2.0220 | **+1.6778** |
| **2** | **Adam Zampa** | 215 | +1.057 | +4.70 | +1.7953 | **+1.6150** |
| **3** | **Jofra Archer** | 325 | +0.875 | +3.87 | +1.5123 | **+1.4083** |
| **4** | **Adam Milne** | 142 | +0.874 | +4.49 | +1.2655 | **+1.0825** |
| **5** | **Wanindu Hasaranga** | 286 | +1.137 | +7.25 | +1.0615 | **+0.9794** |
| **6** | **Akash Madhwal** | 137 | +0.594 | +2.58 | +1.0788 | **+0.9180** |
| **7** | **Trent Boult** | 273 | +0.996 | +6.34 | +0.9472 | **+0.8706** |
| **8** | **Yuzvendra Chahal** | 401 | +1.098 | +7.43 | +0.8604 | **+0.8118** |
| **9** | **Maheesh Theekshana** | 350 | +0.795 | +5.06 | +0.7774 | **+0.7275** |
| **10** | **Corbin Bosch** | 89 | +0.968 | +7.06 | +0.5708 | **+0.4495** |

---

## 5. Execution Instructions

Run all pipeline scripts from the repository root:

```bash
# 1. Exploratory data analysis & player reconciliation
python backend/exploratory_analysis.py

# 2. Win probability model training, calibration & swing threshold estimation
python backend/win_probability_model.py

# 3. Pressure metrics computation, dismissal penalties & top 10 rankings
python backend/pressure_metrics.py
```

### Viewing the Interactive Dashboard
Open `index.html` (or `frontend/index.html`) directly in your web browser, or serve it via GitHub Pages.

---

## License & Competition Notice

Data pack for competition use only. Not for publication.
© Rajasthan Royals • SupeRR Selector Season 2.

# Football xG Model

> Building an Expected Goals model from scratch using StatsBomb open data — replicating what elite football clubs use internally

[![Open in Streamlit](https://john-xg-model.streamlit.app/)]

---

## Overview

This project builds an xG (Expected Goals) model from scratch using
StatsBomb open data from La Liga 2020/21. The model assigns each shot
a probability between 0 and 1 reflecting its likelihood of resulting
in a goal based on shot location and context.

**Key finding:** Goals are scored from an average of 12.9 yards —
5.8 yards closer than non-goals. Distance is the strongest predictor
of shot quality with a logistic regression coefficient of -1.292.

---

## Model Performance

| Model               | AUC-ROC   | Brier Score |
| ------------------- | --------- | ----------- |
| Logistic Regression | **0.772** | 0.101       |
| XGBoost             | 0.711     | 0.109       |
| StatsBomb Baseline  | 0.825     | 0.094       |

Our model achieves AUC of 0.772 — only 0.053 behind StatsBomb's
professional baseline. The gap is attributable to additional features
StatsBomb uses (goalkeeper position, defensive shape) not available
in the open dataset.

---

## Key Findings

### Shot Distance vs Conversion Rate

| Distance    | Conversion Rate |
| ----------- | --------------- |
| 0-5 yards   | 60.0%           |
| 5-10 yards  | 23.8%           |
| 10-15 yards | 19.7%           |
| 20-25 yards | 6.7%            |
| 40+ yards   | 0.0%            |

### Feature Importance

| Feature        | Coefficient | Meaning                  |
| -------------- | ----------- | ------------------------ |
| Distance       | -1.292      | Strongest predictor      |
| Is Header      | -0.401      | Headers score less often |
| Angle          | -0.374      | Wide angles reduce xG    |
| Under Pressure | -0.313      | Pressure reduces xG      |

### Player xG Analysis

| Player      | Goals | xG    | Goals - xG |
| ----------- | ----- | ----- | ---------- |
| Messi       | 30    | 22.86 | **+7.14**  |
| Griezmann   | 12    | 10.90 | +1.10      |
| Braithwaite | 2     | 5.05  | **-3.05**  |

---

## Tech Stack

| Tool          | Purpose                       |
| ------------- | ----------------------------- |
| Python        | Core language                 |
| StatsBombpy   | Football event data API       |
| mplsoccer     | Football pitch visualisations |
| Pandas, NumPy | Data manipulation             |
| Scikit-learn  | Logistic Regression xG model  |
| XGBoost       | Advanced model comparison     |
| Streamlit     | Interactive dashboard         |

---

## Project Structure

    football-xg-model/
    ├── data/
    │   └── shots_features.csv
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_visualisation.ipynb
    │   ├── 04_xg_model.ipynb
    │   └── 05_player_xg_analysis.ipynb
    ├── reports/
    │   ├── shot_map.png
    │   ├── goal_probability_heatmap.png
    │   ├── distance_vs_conversion.png
    │   ├── roc_curve_xg.png
    │   ├── goals_vs_xg.png
    │   └── messi_shot_map.png
    ├── dashboard/
    │   └── app.py
    ├── requirements.txt
    └── README.md

---

## Installation

```bash
git clone https://github.com/John-Ayomide/football-xg-model
cd football-xg-model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Author

**John Ayomide**

- GitHub: [@John-Ayomide](https://github.com/John-Ayomide)
- LinkedIn: [https://www.linkedin.com/in/john-aiyenomuro-19aa26211/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BOjRuavaNSE%2Bbrrf03mDzUA%3D%3D](https://linkedin.com/in/yourprofile)

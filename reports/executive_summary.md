# Executive Summary — Football xG Model

**Author:** John Ayomide
**Data:** StatsBomb Open Data — La Liga 2020/21
**Tools:** Python, StatsBombpy, mplsoccer, Scikit-learn, Streamlit

## Objective

Build an Expected Goals model from scratch that assigns each
shot a probability of resulting in a goal based on shot quality
features — replicating the methodology used by elite football clubs.

## Key Findings

1. Goals scored from avg 12.9 yards vs 18.7 for non-goals
2. 0-5 yard shots convert at 60% — 40+ yard shots at 0%
3. Distance is the strongest xG predictor (coeff: -1.292)
4. Messi: +7.14 goals above xG — statistically exceptional

## Model Performance

- Logistic Regression AUC: 0.772
- StatsBomb baseline AUC: 0.825
- Gap: 0.053 (attributable to hidden features)

## Deliverables

- 5 Jupyter notebooks with full methodology
- 6 professional visualisations including pitch maps
- Live Streamlit dashboard with xG predictor
- Full GitHub repository

# ============================================
# FOOTBALL xG MODEL DASHBOARD
# Interactive Expected Goals Analysis
# La Liga 2020/21 — StatsBomb Open Data
# Author: John Ayomide
# ============================================

import os
from pathlib import Path

# Works both locally and on Streamlit Cloud
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mplsoccer import Pitch, VerticalPitch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="Football xG Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }
    [data-testid="metric-container"] {
        background: #111118;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #6b6b80 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00e5a0 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    h1, h2, h3 { color: #e8e8f0 !important; }
    hr { border-color: #2a2a3a !important; }
    .stSlider label { color: #9999aa !important; }
    .stSelectbox label { color: #9999aa !important; }
    .stButton button {
        background: linear-gradient(135deg, #00e5a0, #7b61ff) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100%;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load and prepare data ────────────────────
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / 'data' / 'shots_features.csv'
    return pd.read_csv(data_path)

@st.cache_resource
def train_model(df):
    feature_cols = ['distance', 'angle', 'is_header',
                    'under_pressure', 'is_open_play']
    X = df[feature_cols]
    y = df['is_goal']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, feature_cols

shots = load_data()
model, scaler, feature_cols = train_model(shots)

# Generate xG for all shots
shots['our_xg'] = model.predict_proba(
    scaler.transform(shots[feature_cols])
)[:, 1]

# ── Header ───────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
    <span style='font-size:11px;color:#00e5a0;
    letter-spacing:3px;text-transform:uppercase'>
    Portfolio Project 02 — Football Analytics</span>
</div>
<h1 style='font-size:32px;font-weight:700;
color:#e8e8f0;margin-bottom:8px'>
    Football <span style='color:#00e5a0'>xG Model</span>
</h1>
<p style='color:#6b6b80;font-size:15px;
margin-bottom:32px'>
    Expected Goals analysis — La Liga 2020/21 ·
    StatsBomb open data · Logistic Regression
    (AUC 0.772)
</p>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Shots", f"{len(shots):,}")
with k2:
    st.metric("Total Goals", f"{shots['is_goal'].sum():,}")
with k3:
    st.metric("Conversion Rate",
              f"{shots['is_goal'].mean():.1%}")
with k4:
    st.metric("Model AUC", "0.772")

st.markdown("---")

# ── Tabs ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚽ Shot Map",
    "👤 Player Analysis",
    "🔮 xG Predictor"
])

# ── TAB 1: SHOT MAP ──────────────────────────
with tab1:
    st.markdown("### Shot Map — All 839 Shots")
    st.markdown(
        "<p style='color:#6b6b80;font-size:14px'>"
        "Green = Goal · Red = No Goal · "
        "Larger dot = higher xG value</p>",
        unsafe_allow_html=True
    )

    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#111118',
        line_color='#2a2a3a',
        half=True
    )
    fig, ax = pitch.draw(figsize=(12, 7))
    fig.patch.set_facecolor('#111118')

    non_goals = shots[shots['is_goal']==0]
    goals = shots[shots['is_goal']==1]

    pitch.scatter(
        non_goals['x'], non_goals['y'], ax=ax,
        s=non_goals['our_xg'] * 500 + 20,
        color='#ff6b6b', alpha=0.4,
        edgecolors='#ff6b6b', linewidth=0.3,
        label=f'No Goal ({len(non_goals)})'
    )
    pitch.scatter(
        goals['x'], goals['y'], ax=ax,
        s=goals['our_xg'] * 500 + 40,
        color='#00e5a0', alpha=0.85,
        edgecolors='white', linewidth=0.5,
        label=f'Goal ({len(goals)})'
    )
    ax.legend(loc='lower left',
              facecolor='#1a1a24',
              labelcolor='#e8e8f0', fontsize=10)

    st.pyplot(fig)

    # Distance chart
    st.markdown("### Conversion Rate by Distance")
    shots['distance_band'] = pd.cut(
        shots['distance'],
        bins=[0,5,10,15,20,25,30,40,100],
        labels=['0-5','5-10','10-15','15-20',
                '20-25','25-30','30-40','40+']
    )
    conv = shots.groupby(
        'distance_band', observed=True
    )['is_goal'].agg(['mean','count']).reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#111118')
    ax2.set_facecolor('#111118')
    bars = ax2.bar(conv['distance_band'],
                   conv['mean']*100,
                   color='#00e5a0', alpha=0.8,
                   edgecolor='#2a2a3a')
    for bar, (_, row) in zip(bars, conv.iterrows()):
        ax2.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f"{row['mean']*100:.1f}%",
            ha='center', va='bottom',
            color='#e8e8f0', fontsize=9
        )
    ax2.set_xlabel('Distance (yards)',
                   color='#6b6b80')
    ax2.set_ylabel('Conversion Rate (%)',
                   color='#6b6b80')
    ax2.tick_params(colors='#6b6b80')
    ax2.spines['bottom'].set_color('#2a2a3a')
    ax2.spines['left'].set_color('#2a2a3a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# ── TAB 2: PLAYER ANALYSIS ───────────────────
with tab2:
    st.markdown("### Player xG Analysis")

    # Build player stats
    player_stats = shots.groupby('player').agg(
        shots=('is_goal','count'),
        goals=('is_goal','sum'),
        xg=('our_xg','sum')
    ).reset_index()
    player_stats['goals_minus_xg'] = (
        player_stats['goals'] - player_stats['xg']
    )
    player_stats = player_stats[
        player_stats['shots'] >= 5
    ].sort_values('xg', ascending=False)

    # Goals vs xG scatter
    st.markdown("#### Goals vs Expected Goals")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('#111118')
    ax3.set_facecolor('#111118')

    colors = ['#00e5a0' if g > 0 else '#ff6b6b'
              for g in player_stats['goals_minus_xg']]

    ax3.scatter(
        player_stats['xg'],
        player_stats['goals'],
        c=colors,
        s=player_stats['shots']*2,
        alpha=0.7,
        edgecolors='white', linewidth=0.5
    )

    max_val = max(player_stats['xg'].max(),
                  player_stats['goals'].max()) + 1
    ax3.plot([0,max_val],[0,max_val],
             '--', color='#6b6b80', linewidth=1.5)

    for _, row in player_stats.nlargest(
        6,'shots'
    ).iterrows():
        name = row['player'].split()[-1]
        ax3.annotate(name,
                     (row['xg'], row['goals']),
                     textcoords="offset points",
                     xytext=(8,4),
                     fontsize=8,
                     color='#e8e8f0')

    ax3.set_xlabel('Expected Goals (xG)',
                   color='#6b6b80')
    ax3.set_ylabel('Actual Goals', color='#6b6b80')
    ax3.tick_params(colors='#6b6b80')
    ax3.spines['bottom'].set_color('#2a2a3a')
    ax3.spines['left'].set_color('#2a2a3a')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

    # Player selector — shot map
    st.markdown("#### Individual Player Shot Map")
    players_list = sorted(
        shots[shots.groupby('player')[
            'player'
        ].transform('count') >= 5]['player'].unique()
    )
    selected = st.selectbox(
        "Select a player", players_list
    )

    if selected:
        p_shots = shots[shots['player']==selected]
        p_goals = p_shots[p_shots['is_goal']==1]
        p_miss = p_shots[p_shots['is_goal']==0]

        pitch2 = Pitch(
            pitch_type='statsbomb',
            pitch_color='#111118',
            line_color='#2a2a3a',
            half=True
        )
        fig4, ax4 = pitch2.draw(figsize=(10, 6))
        fig4.patch.set_facecolor('#111118')

        pitch2.scatter(
            p_miss['x'], p_miss['y'], ax=ax4,
            s=80, color='#ff6b6b', alpha=0.5,
            edgecolors='#ff6b6b', linewidth=0.3,
            label=f'No Goal ({len(p_miss)})'
        )
        pitch2.scatter(
            p_goals['x'], p_goals['y'], ax=ax4,
            s=120, color='#00e5a0', alpha=0.9,
            edgecolors='white', linewidth=0.5,
            label=f'Goal ({len(p_goals)})'
        )

        xg_total = p_shots['our_xg'].sum()
        g_minus_xg = len(p_goals) - xg_total
        sign = '+' if g_minus_xg >= 0 else ''

        ax4.set_title(
            f"{selected.split()[-1]} — "
            f"{len(p_shots)} shots · "
            f"{len(p_goals)} goals · "
            f"xG: {xg_total:.2f} · "
            f"G-xG: {sign}{g_minus_xg:.2f}",
            color='#e8e8f0', fontsize=12,
            fontweight='bold', pad=15
        )
        ax4.legend(
            loc='lower left',
            facecolor='#1a1a24',
            labelcolor='#e8e8f0', fontsize=9
        )
        st.pyplot(fig4)

        # Player stats table
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shots", len(p_shots))
        with col2:
            st.metric("Goals", len(p_goals))
        with col3:
            st.metric("xG", f"{xg_total:.2f}")
        with col4:
            st.metric("Goals - xG",
                      f"{sign}{g_minus_xg:.2f}")

# ── TAB 3: xG PREDICTOR ──────────────────────
with tab3:
    st.markdown("### xG Predictor")
    st.markdown(
        "<p style='color:#6b6b80;font-size:14px'>"
        "Enter shot details to calculate the "
        "expected goal probability.</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        distance = st.slider(
            "Distance from goal (yards)", 1, 40, 15
        )
        angle = st.slider(
            "Shot angle (degrees)", 0, 90, 30
        )
        is_header = st.selectbox(
            "Shot type", ["Foot", "Header"]
        )

    with col2:
        under_pressure = st.selectbox(
            "Under pressure?", ["No", "Yes"]
        )
        is_open_play = st.selectbox(
            "From open play?", ["Yes", "No"]
        )

    if st.button("⚽ Calculate xG"):
        input_data = pd.DataFrame([{
            'distance': distance,
            'angle': angle,
            'is_header': 1 if is_header=="Header" else 0,
            'under_pressure': 1 if under_pressure=="Yes" else 0,
            'is_open_play': 1 if is_open_play=="Yes" else 0
        }])

        input_scaled = scaler.transform(input_data)
        xg_value = float(
            model.predict_proba(input_scaled)[0][1]
        )

        # Display result
        if xg_value >= 0.3:
            st.markdown(f"""
            <div style='background:rgba(0,229,160,0.1);
            border:1px solid rgba(0,229,160,0.4);
            border-radius:12px;padding:20px;
            margin-top:16px'>
                <div style='font-size:22px;
                font-weight:700;color:#00e5a0'>
                    🟢 HIGH QUALITY CHANCE
                </div>
                <div style='font-size:36px;
                font-weight:700;color:#00e5a0;
                margin:8px 0'>xG = {xg_value:.3f}</div>
                <div style='color:#9999aa;
                font-size:14px'>
                    This shot has a {xg_value:.1%}
                    probability of resulting in a goal
                </div>
            </div>""", unsafe_allow_html=True)

        elif xg_value >= 0.1:
            st.markdown(f"""
            <div style='background:rgba(255,209,102,0.1);
            border:1px solid rgba(255,209,102,0.4);
            border-radius:12px;padding:20px;
            margin-top:16px'>
                <div style='font-size:22px;
                font-weight:700;color:#ffd166'>
                    🟡 MODERATE CHANCE
                </div>
                <div style='font-size:36px;
                font-weight:700;color:#ffd166;
                margin:8px 0'>xG = {xg_value:.3f}</div>
                <div style='color:#9999aa;
                font-size:14px'>
                    This shot has a {xg_value:.1%}
                    probability of resulting in a goal
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:rgba(255,107,107,0.1);
            border:1px solid rgba(255,107,107,0.4);
            border-radius:12px;padding:20px;
            margin-top:16px'>
                <div style='font-size:22px;
                font-weight:700;color:#ff6b6b'>
                    🔴 LOW QUALITY CHANCE
                </div>
                <div style='font-size:36px;
                font-weight:700;color:#ff6b6b;
                margin:8px 0'>xG = {xg_value:.3f}</div>
                <div style='color:#9999aa;
                font-size:14px'>
                    This shot has a {xg_value:.1%}
                    probability of resulting in a goal
                </div>
            </div>""", unsafe_allow_html=True)

        # Show distance context
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1a1a24;
        border:1px solid #2a2a3a;
        border-radius:8px;padding:16px'>
            <div style='font-size:13px;
            color:#6b6b80;margin-bottom:8px'>
                Shot context
            </div>
            <div style='font-size:13px;
            color:#9999aa;line-height:1.8'>
                📍 Distance: {distance} yards from goal<br>
                📐 Angle: {angle}° from centre<br>
                🦶 Technique: {is_header}<br>
                🛡️ Pressure: {under_pressure}<br>
                ⚽ Play type: 
                {"Open play" if is_open_play=="Yes" 
                 else "Set piece"}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='color:#6b6b80;font-size:12px;
text-align:center'>
    Built by <strong style='color:#e8e8f0'>
    John Ayomide</strong> &nbsp;·&nbsp;
    <a href='https://github.com/John-Ayomide/
football-xg-model'
    style='color:#00e5a0;text-decoration:none'>
    GitHub</a> &nbsp;·&nbsp;
    Data: StatsBomb Open Data · La Liga 2020/21
</div>
""", unsafe_allow_html=True)
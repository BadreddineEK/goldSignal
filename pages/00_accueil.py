"""
00_accueil.py — Page d'accueil GoldSignal.

Onboarding, présentation du projet, cours du jour, guide de navigation.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Spots temps réel (avec fallback silencieux)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _get_spots():
    try:
        from data.fetcher import get_spot_xau_eur, get_spot_xag_eur
        return get_spot_xau_eur(), get_spot_xag_eur()
    except Exception:
        return None, None

xau_g, xag_g = _get_spots()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <div style="font-size: 3rem;">🥇</div>
        <h1 style="font-size: 2.4rem; font-weight: 800; margin: 0.3rem 0;">GoldSignal</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; margin: 0;">
            Analyse & aide à la décision pour métaux précieux physiques
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Spots du jour
# ---------------------------------------------------------------------------
col_xau, col_xag, col_ratio, col_nap = st.columns(4)

try:
    from utils.formatting import fmt_eur
    col_xau.metric("🥇 Or / g fin",    fmt_eur(xau_g) if xau_g else "—",  help="Cours spot GC=F en €/g")
    col_xag.metric("🥈 Argent / g fin", fmt_eur(xag_g, 4) if xag_g else "—", help="Cours spot SI=F en €/g")
    if xau_g and xag_g:
        ratio = xau_g / xag_g
        col_ratio.metric("⚖️ Ratio Or/Argent", f"{ratio:.0f}", help="Combien de g d'argent pour 1 g d'or")
    if xau_g:
        nap20 = xau_g * 5.806
        col_nap.metric("🪙 Napoléon 20F spot", fmt_eur(nap20, 0), help="Valeur intrinsèque or pur — hors prime")
except Exception:
    col_xau.metric("🥇 Or / g fin", "—")
    col_xag.metric("🥈 Argent / g fin", "—")

st.caption(f"Cours indicatifs yfinance — {datetime.now().strftime('%d/%m/%Y %H:%M')} (cache 1h)")

# Signal ML si disponible en session
if st.session_state.get("latest_signal") is not None:
    sig = int(st.session_state["latest_signal"])
    horizon = st.session_state.get("horizon", 5)
    sig_map = {1: ("🟢", "Haussier", "success"), 0: ("⚪", "Neutre", "info"), -1: ("🔴", "Baissier", "warning")}
    emoji, label, kind = sig_map.get(sig, ("⚪", "Neutre", "info"))
    getattr(st, kind)(f"{emoji} **Signal ML actuel : {label}** à horizon {horizon}j — entraîné sur la page Prédictions ML")

st.markdown("---")

# ---------------------------------------------------------------------------
# Présentation du projet
# ---------------------------------------------------------------------------
st.subheader("📌 À propos de GoldSignal")

col_desc, col_stack = st.columns([3, 2])

with col_desc:
    st.markdown("""
    **GoldSignal** est un outil d'analyse pour l'investisseur en métaux précieux physiques.
    Il combine trois dimensions complémentaires :

    1. **🧮 Évaluation terrain** — Est-ce que le prix affiché par un comptoir est juste ?
       Calcul prime, spread, score de qualité en temps réel.

    2. **📊 Contexte macro & technique** — Dollar, taux réels, VIX, saisonnalité, corrélations.
       Comprendre les forces qui font bouger les cours.

    3. **🤖 Intelligence artificielle** — Prédictions de tendance à 5/15/30 jours via
       Random Forest, XGBoost, LSTM et un méta-apprenant hybride entraîné en walk-forward
       strict (zéro data leakage).

    > L'objectif pédagogique : **montrer une chaîne ML complète**, de la feature engineering
    > jusqu'au backtesting P&L, appliquée à un cas concret.
    """)

with col_stack:
    st.markdown("**Stack technique**")
    stack_items = [
        ("Python 3.13",       "Langage principal"),
        ("Streamlit 1.50",    "Interface web"),
        ("scikit-learn",      "Random Forest, walk-forward CV"),
        ("XGBoost",           "Gradient boosting"),
        ("PyTorch",           "LSTM bidirectionnel"),
        ("statsmodels",       "ARIMA, tests ADF/DM"),
        ("yfinance + FRED",   "Données marché"),
        ("Plotly",            "Visualisations"),
        ("SQLite",            "Config persistante"),
    ]
    for lib, desc in stack_items:
        st.markdown(f"- **`{lib}`** — {desc}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Guide de navigation
# ---------------------------------------------------------------------------
st.subheader("🗺️ Par où commencer ?")
st.caption("Flux recommandé pour explorer l'application de façon cohérente.")

steps = [
    {
        "icon": "📊",
        "page": "Macro & Technique",
        "action": "Regarder le **contexte macro actuel**",
        "detail": "Score macro global, cours historique Or, corrélations DXY/taux réels, saisonnalité.",
        "time": "2 min",
        "level": "🟢 Facile",
    },
    {
        "icon": "🧮",
        "page": "Calculateur",
        "action": "**Évaluer un prix comptoir** terrain",
        "detail": "Saisissez Ask/Bid pour une pièce (Napoléon 20F, Britannia…) → prime%, spread%, verdict.",
        "time": "1 min",
        "level": "🟢 Facile",
    },
    {
        "icon": "🤖",
        "page": "Prédictions ML",
        "action": "Charger ou entraîner les **modèles ML**",
        "detail": "Chargez un modèle pré-entraîné en 1 clic ou lancez l'entraînement complet (RF+XGB+LSTM). Horizons : 5j / 15j / 30j.",
        "time": "1 min (pré-entraîné) · 3-8 min (entraînement)",
        "level": "🟡 Intermédiaire",
    },
    {
        "icon": "💰",
        "page": "Simulateur",
        "action": "**Simuler un investissement**",
        "detail": "Que se serait-il passé si j'avais investi X€ le [date] ? Est-ce le bon moment ? Projection Monte-Carlo.",
        "time": "2 min",
        "level": "🟢 Facile",
    },
    {
        "icon": "📈",
        "page": "Backtesting P&L",
        "action": "Voir la **performance réelle des signaux** ML",
        "detail": "Equity curve, Sharpe, drawdown, comparaison stratégies vs Buy & Hold. Nécessite un modèle entraîné.",
        "time": "2 min",
        "level": "🟡 Intermédiaire",
    },
    {
        "icon": "📐",
        "page": "Benchmark ML",
        "action": "**Comparer les modèles** en détail",
        "detail": "Tableau multi-métriques, radar chart, matrices de confusion, test Diebold-Mariano, calibration.",
        "time": "5 min",
        "level": "🔴 Avancé",
    },
]

for i, step in enumerate(steps, 1):
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 6, 2])
        with c1:
            st.markdown(f"<div style='text-align:center;font-size:1.8rem;padding-top:4px'>{step['icon']}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Étape {i} — {step['page']}**")
            st.markdown(f"{step['action']}")
            st.caption(step["detail"])
        with c3:
            st.caption(f"⏱ {step['time']}")
            st.caption(step["level"])

st.markdown("---")

# ---------------------------------------------------------------------------
# Méthodologie ML en résumé
# ---------------------------------------------------------------------------
st.subheader("🔬 Méthodologie ML — Points clés")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("""
    **Walk-Forward CV**
    
    Les modèles sont évalués en *walk-forward* strict :
    entraînement sur passé uniquement, test sur futur inconnu.
    Aucune donnée future ne fuit dans l'entraînement
    (**zéro data leakage**).
    """)

with col_m2:
    st.markdown("""
    **4 modèles + benchmark**
    
    - ARIMA (baseline statistique)
    - Random Forest
    - XGBoost
    - LSTM bidirectionnel (PyTorch)
    - Meta-apprenant hybride (stacking)
    - Random Walk (benchmark naïf)
    """)

with col_m3:
    st.markdown("""
    **Métriques rigoureuses**
    
    - Directional Accuracy (DA%)
    - Brier Score, Log-Loss
    - Test Diebold-Mariano (significativité)
    - Sharpe / Sortino / Calmar
    - Maximum Drawdown
    - Win Rate par classe
    """)

st.markdown("---")

# ---------------------------------------------------------------------------
# Avertissement légal
# ---------------------------------------------------------------------------
st.info(
    "⚠️ **Avertissement** — GoldSignal est un outil pédagogique et d'aide à la réflexion. "
    "Il ne constitue pas un conseil en investissement. Les performances passées ne préjugent pas "
    "des performances futures. Tout investissement comporte un risque de perte en capital."
)

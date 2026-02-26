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
_raw_sig = st.session_state.get("latest_signal")
if _raw_sig is not None:
    # latest_signal peut être un int {-1,0,1} ou un dict {"signal": int, ...}
    if isinstance(_raw_sig, dict):
        _raw_sig = _raw_sig.get("signal", _raw_sig.get("direction", _raw_sig.get("value")))
    try:
        sig = int(_raw_sig)
    except (TypeError, ValueError):
        sig = None
    horizon = st.session_state.get("horizon", 5)
    if sig is not None:
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
st.caption("Aperçu de la chaîne Data Science utilisée — du prix brut au signal de trading opérationnel.")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    with st.container(border=True):
        st.markdown("**🔄 Walk-Forward Cross-Validation**")
        st.markdown("""
        Contrairement à une `train_test_split` classique, la walk-forward
        divise les données en **fenêtres temporelles successives** :
        - Fold 1 : train 2019-2021 → test 2022
        - Fold 2 : train 2019-2022 → test 2023
        - etc.
        
        Résultat : **zéro look-ahead bias** — les métriques reflètent
        une vraie performance hors-échantillon, comme en conditions réelles.
        """)

with col_m2:
    with st.container(border=True):
        st.markdown("**🤖 6 modèles en concurrence**")
        st.markdown("""
        | Modèle | Type | Force |
        |---|---|---|
        | Random Walk | Naïf | Benchmark |
        | ARIMA | Stat. | Série temporelle |
        | Random Forest | Ensemble | Robuste, interprétable |
        | XGBoost | Boosting | Performant, rapide |
        | LSTM bidir. | Deep learning | Séquences longues |
        | Stacking | Méta-apprenant | Combine les 4 |
        
        La cible prédite est la **direction** à N jours : 🔴 Baissier / ⚪ Neutre / 🟢 Haussier.
        """)

with col_m3:
    with st.container(border=True):
        st.markdown("**📐 Features engineering**")
        st.markdown("""
        Plus de **40 features** calculées sur le cours Or + contexte macro :
        - *Prix* : log-rendements, volatilité réalisée, SMA 20/50/200j
        - *Momentum* : RSI, MACD, Williams %R, CCI
        - *Macro* : DXY (dollar), taux réels US, ratio Or/Argent
        - *Risque* : VIX, spread 10Y-2Y, momentum SP500
        - *Saisonnalité* : mois, jour de semaine (encodé cyclique)
        
        Toutes les features sont **normalisées** séparément sur chaque fenêtre
        de train (évite la fuite d'information future).
        """)

with st.expander("📊 Pourquoi évaluer avec DA%, Brier et Sharpe — et pas juste l'accuracy ?"):
    st.markdown("""
    #### Directional Accuracy (DA%)
    L'*accuracy* classique comptabilise une erreur identique que le modèle prédise
    Neutre au lieu de Haussier ou Baissier au lieu de Haussier.
    La **DA%** mesure spécifiquement le **sens prédit vs sens réalisé**.
    Un DA% > 55% sur données hors-échantillon est considéré comme économiquement utile
    (benchmarks académiques sur matières premières).

    #### Brier Score
    Mesure la **qualité des probabilités** (et non juste du label final).
    Formule : $BS = \\frac{1}{N}\\sum_{t=1}^{N}(p_t - y_t)^2$ où $p_t$ est la probabilité
    prédite et $y_t$ ∈ {0,1}. **Plus bas = meilleur** (0 = parfait, 1 = catastrophique).
    Baseline naïve : ~0.67 (classes équiprobables). Un BS < 0.55 indique une
    calibration utile.

    #### Sharpe Ratio (backtesting P&L)
    Mesure le **rendement ajusté au risque** : $Sharpe = \\frac{R_{strat} - R_f}{\\sigma_{strat}} \\times \\sqrt{252}$
    - < 0.5 : stratégie risquée
    - 0.5 – 1.0 : acceptable
    - > 1.0 : bon — rare sur données OOS
    - > 2.0 : excellent (suspect si trop beau)

    #### Alpha
    Surperformance **annualisée** de la stratégie vs Buy & Hold (détenir de l'or en continu).
    Un alpha positif signifie que les signaux ML ajoutent de la valeur au-delà d'une
    détention passive.

    #### Test Diebold-Mariano
    Test statistique qui évalue si **deux modèles sont significativement différents**
    en termes d'erreur de prédiction (H₀ : performances identiques). p < 0.05 = différence
    statistiquement significative.
    """)

st.markdown("---")

# ---------------------------------------------------------------------------
# Lexique rapide des termes clés
# ---------------------------------------------------------------------------
with st.expander("📖 Lexique — Termes financiers & ML utilisés dans l'application"):
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("""
        **📈 Termes marchés**

        **Cours spot** : prix de marché en temps réel (hors prime).

        **Prime** : supplément payé au-dessus de la valeur or pur.
        Exemple : Napoléon 20F à 350€ alors que l'or pur vaut 310€ → prime = +13%.

        **Spread** : écart entre prix de vente (ask) et prix d'achat (bid) du comptoir.
        Plus il est faible, meilleur est le prix.

        **CAGR** *(Compound Annual Growth Rate)* : taux de croissance annuel composé.
        CAGR = (valeur_finale / valeur_initiale)^(1/n_années) - 1.

        **RSI 14j** *(Relative Strength Index)* : oscillateur 0-100.
        < 30 = survente (opportunité potentielle) · > 70 = surachat.
        Formule : RSI = 100 - 100/(1 + RS) où RS = moyenne gains / moyenne pertes sur 14j.

        **Bandes de Bollinger** : cours ± 2×écart-type sur 20j.
        Cours en bas de bande → prix relativement bas sur la période.

        **Percentile 1 an** : rang du prix actuel parmi les 252 derniers jours de bourse.
        Percentile 80 = prix plus élevé que 80% des jours de l'année.

        **SMA (Simple Moving Average)** : moyenne mobile simple.
        Prix > SMA200 → tendance haussière long terme.
        """)
    with col_l2:
        st.markdown("""
        **🤖 Termes Machine Learning**

        **OOS / Hors-échantillon** : données **non vues** pendant l'entraînement.
        Toutes les métriques affichées sont calculées OOS.

        **Walk-forward** : validation temporelle glissante — évite de "voir le futur".

        **Directional Accuracy (DA%)** : % de fois où le signe prédit (↑/↓) correspond
        au signe réalisé. Baseline naïve : ~50%.

        **Brier Score** : erreur quadratique sur les probabilités prédites (0=parfait).

        **Log-Loss** : log-vraisemblance négative — pénalise les prédictions
        très confiantes mais fausses.

        **Conviction** : max(P_haussier, P_baissier) × 2 - 1, normalisé 0-100%.
        Mesure l'assurance du modèle dans sa prediction.

        **Stacking / Méta-apprenant** : modèle qui prend les **prédictions des autres
        modèles comme features** pour produire une prédiction combinée.

        **Max Drawdown** : pire perte depuis un sommet du capital.
        Exemple : capital passe de 10 000€ à 8 000€ → drawdown = -20%.

        **Sharpe Ratio** : rendement annualisé / volatilité annualisée.
        Mesure le rendement "par unité de risque".

        **Alpha** : performance de la stratégie ML **au-delà** de Detroit de l'or.
        Alpha positif = les signaux ajoutent de la valeur.
        """)
# Avertissement légal
# ---------------------------------------------------------------------------
st.info(
    "⚠️ **Avertissement** — GoldSignal est un outil pédagogique et d'aide à la réflexion. "
    "Il ne constitue pas un conseil en investissement. Les performances passées ne préjugent pas "
    "des performances futures. Tout investissement comporte un risque de perte en capital."
)

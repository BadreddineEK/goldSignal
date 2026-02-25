"""
07_backtest.py — Simulation P&L des signaux ML (GoldSignal).

Lit les résultats d'entraînement (session_state ou DB) et simule une stratégie
long-only basée sur les prédictions out-of-sample.

Sections :
  1. Résumé des métriques clés (Sharpe, MDD, Win Rate, Alpha)
  2. Équity curve stratégie vs Buy & Hold
  3. Drawdown rolling
  4. Distribution des rendements par trade
  5. Tableau comparatif de toutes les stratégies
  6. Calendrier mensuel des rendements (heatmap mois × année)
  7. Analyse des signaux : fréquence, précision par classe
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models.pl_simulator import (
    simulate_strategy,
    compute_pl_metrics,
    extract_signals_from_ml,
    extract_signals_from_lstm,
    extract_signals_from_hybrid,
    build_strategy_comparison,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.header("📈 Backtesting P&L")
st.caption(
    "Simulation de stratégie long-only sur les signaux out-of-sample (walk-forward zéro-leakage). "
    "Capital initial : **10 000 €** — frais : **0.1% par trade**."
)

# ---------------------------------------------------------------------------
# Chargement des données de session
# ---------------------------------------------------------------------------
ml_results   = st.session_state.get("ml_results")
log_returns  = st.session_state.get("_backtest_log_returns")
arima_result = st.session_state.get("arima_result", {})
lstm_result  = st.session_state.get("lstm_result", {})
hybrid_result= st.session_state.get("hybrid_result", {})
horizon      = st.session_state.get("horizon", 5)

# Essayer de récupérer log_returns depuis les données marché en cache
if log_returns is None:
    try:
        from pages.utils_shared import load_log_returns  # type: ignore
    except ImportError:
        pass

if ml_results is None:
    st.warning(
        "⚠️ Aucun modèle entraîné. Rendez-vous sur **🤖 Prédictions ML** et lancez l'entraînement."
    )
    st.stop()

# Récupérer les vrais log-rendements prix depuis session_state
# (stockés par 03_predictions.py comme np.log(close).diff(1) — PAS les labels {-1,0,+1})
log_returns_oos = log_returns  # chargé ligne ~50 depuis session_state["_backtest_log_returns"]

if log_returns_oos is None or (isinstance(log_returns_oos, pd.Series) and log_returns_oos.empty):
    # Fallback : reconstruire depuis les données marché si disponible
    try:
        from data.fetcher import fetch_ticker
        from analysis.features import ohlcv_usd_oz_to_eur_g, compute_log_returns as _clr
        from data.database import get_config as _gc
        _tickers = _gc("tickers_yfinance") or {}
        _df_xau = fetch_ticker(_tickers.get("xau", "GC=F"))
        _df_eur = fetch_ticker(_tickers.get("eurusd", "EURUSD=X"))
        _df_xau_eur = ohlcv_usd_oz_to_eur_g(_df_xau, _df_eur)
        _close_fb = _df_xau_eur["close"].dropna()
        log_returns_oos = _clr(_close_fb, [1])["log_return_1"].dropna()
        st.caption("ℹ️ Log-returns reconstruits depuis les données marché.")
    except Exception as _fb_e:
        st.warning(f"Les log-returns prix ne sont pas disponibles — relancez l'entraînement sur **Prédictions ML**. ({_fb_e})")
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar : paramètres de simulation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation P&L")
    initial_capital = st.number_input("Capital initial (€)", 1000, 1_000_000, 10_000, 1000)
    cost_bps = st.slider("Frais aller-retour (bps)", 0, 50, 10, 5,
                          help="1 bps = 0.01%. Gold physique : ~20-50 bps spread.")
    cost_pct = cost_bps / 10_000

# ---------------------------------------------------------------------------
# Construction des signaux pour chaque modèle disponible
# ---------------------------------------------------------------------------
models_signals: dict[str, pd.Series] = {}

if ml_results.get("rf", {}).get("predictions") is not None:
    s = extract_signals_from_ml(ml_results, "rf")
    if not s.empty:
        models_signals["Random Forest"] = s

if ml_results.get("xgb", {}).get("predictions") is not None:
    s = extract_signals_from_ml(ml_results, "xgb")
    if not s.empty:
        models_signals["XGBoost"] = s

if lstm_result and lstm_result.get("predictions") is not None:
    s = extract_signals_from_lstm(lstm_result)
    if not s.empty:
        models_signals["LSTM"] = s

if hybrid_result and hybrid_result.get("predictions") is not None:
    s = extract_signals_from_hybrid(hybrid_result)
    if not s.empty:
        models_signals["Hybride"] = s

if not models_signals:
    st.info("Aucun signal ML disponible. Relancez l'entraînement sur la page **Prédictions ML**.")
    st.stop()

# ---------------------------------------------------------------------------
# Simulation pour chaque modèle
# ---------------------------------------------------------------------------
simulations: dict[str, pd.DataFrame] = {}
pl_metrics:  dict[str, dict]         = {}

for model_name, sigs in models_signals.items():
    df_sim = simulate_strategy(sigs, log_returns_oos, initial_capital, cost_pct)
    if not df_sim.empty:
        simulations[model_name] = df_sim
        pl_metrics[model_name]  = compute_pl_metrics(df_sim, horizon)

if not simulations:
    st.error("La simulation a échoué — vérifiez que les signaux et les log-returns sont alignés.")
    st.stop()

# Modèle de référence pour les vues détaillées
model_choices = list(simulations.keys())
selected_model = st.selectbox("Modèle de référence pour les graphiques détaillés", model_choices, index=0)

df_ref  = simulations[selected_model]
met_ref = pl_metrics[selected_model]

st.markdown("---")

# ===========================================================================
# SECTION 1 — Métriques clés
# ===========================================================================
st.subheader("📊 Métriques clés")

c1, c2, c3, c4, c5, c6 = st.columns(6)

def _color_metric(val, good_positive=True):
    if good_positive:
        return "normal" if val >= 0 else "inverse"
    return "inverse" if val >= 0 else "normal"

c1.metric("Rdt Total",
          f"{met_ref['total_return_pct']:+.1f}%",
          f"B&H : {met_ref['bh_total_return_pct']:+.1f}%")
c2.metric("Alpha annualisé",
          f"{met_ref['alpha_pct']:+.2f}%",
          help="Surperformance vs Buy & Hold")
c3.metric("Sharpe (annualisé)",
          f"{met_ref['sharpe']:.3f}",
          help=">1 = bon, >2 = excellent")
c4.metric("Max Drawdown",
          f"{met_ref['max_drawdown_pct']:.1f}%",
          help="Pire perte depuis un sommet")
c5.metric("Win Rate",
          f"{met_ref['win_rate_pct']:.1f}%",
          f"sur {met_ref['n_trades']} trades")
c6.metric("Profit Factor",
          f"{met_ref['profit_factor']:.2f}",
          help=">1.5 = stratégie rentable")

st.markdown("---")

# ===========================================================================
# SECTION 2 — Equity curve + Drawdown
# ===========================================================================
st.subheader("📈 Courbe de capitalisation vs Buy & Hold")

fig_eq = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=["Équity curve", "Drawdown (%)"],
)

COLORS = ["#f59e0b", "#22c55e", "#60a5fa", "#a78bfa", "#fb7185"]

# Toutes les équity curves
for i, (mname, df_s) in enumerate(simulations.items()):
    color = COLORS[i % len(COLORS)]
    lw = 3 if mname == selected_model else 1.5
    dash = "solid" if mname == selected_model else "dot"
    fig_eq.add_trace(
        go.Scatter(x=df_s.index, y=df_s["equity"],
                   name=mname, line=dict(color=color, width=lw, dash=dash)),
        row=1, col=1,
    )

# Buy & Hold
fig_eq.add_trace(
    go.Scatter(x=df_ref.index, y=df_ref["bh_equity"],
               name="Buy & Hold", line=dict(color="#94a3b8", width=1.5, dash="longdash"),
               opacity=0.8),
    row=1, col=1,
)

# Drawdown du modèle de référence
fig_eq.add_trace(
    go.Scatter(
        x=df_ref.index, y=df_ref["drawdown_pct"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
        line=dict(color="#ef4444", width=1),
        name="Drawdown",
    ),
    row=2, col=1,
)
fig_eq.add_hline(y=0, line_dash="dash", line_color="#475569", row=2, col=1)

fig_eq.update_layout(
    height=520,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)",
    legend=dict(orientation="h", y=1.08),
    yaxis=dict(title="Capital (€)", gridcolor="#1e293b"),
    yaxis2=dict(title="Drawdown (%)", gridcolor="#1e293b"),
    xaxis2=dict(gridcolor="#1e293b"),
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_eq, width="stretch")

st.markdown("---")

# ===========================================================================
# SECTION 3 — Distribution des rendements par trade
# ===========================================================================
st.subheader("📦 Distribution des rendements par trade")

# Calculer les rendements agrégés par trade (bloc continu position=1)
pos = df_ref["position"]
trade_groups = (pos != pos.shift()).cumsum()
trade_pl_list = []
trade_dates   = []
for gid, grp in df_ref[df_ref["position"] == 1].groupby(trade_groups[df_ref["position"] == 1]):
    trade_pl_list.append(float(grp["strat_return"].sum() * 100))
    trade_dates.append(grp.index[-1])

if trade_pl_list:
    wins   = [x for x in trade_pl_list if x > 0]
    losses = [x for x in trade_pl_list if x <= 0]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=wins, name=f"Gagnants ({len(wins)})",
        marker_color="#22c55e", opacity=0.8, nbinsx=20,
    ))
    fig_dist.add_trace(go.Histogram(
        x=losses, name=f"Perdants ({len(losses)})",
        marker_color="#ef4444", opacity=0.8, nbinsx=20,
    ))
    fig_dist.add_vline(x=0, line_dash="dash", line_color="#f59e0b")
    fig_dist.update_layout(
        barmode="overlay",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        xaxis_title="Rendement par trade (%)",
        yaxis_title="Fréquence",
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dist, width="stretch")
else:
    st.info("Pas assez de trades pour la distribution.")

st.markdown("---")

# ===========================================================================
# SECTION 4 — Calendrier mensuel des rendements
# ===========================================================================
st.subheader("🗓️ Rendements mensuels (heatmap)")

monthly = df_ref["strat_return"].resample("ME").sum() * 100
if not monthly.empty:
    df_monthly = monthly.reset_index()
    df_monthly.columns = ["date", "ret_pct"]
    df_monthly["year"]  = df_monthly["date"].dt.year.astype(str)
    df_monthly["month"] = df_monthly["date"].dt.strftime("%b")

    _month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot = df_monthly.pivot_table(
        index="year", columns="month", values="ret_pct", aggfunc="sum"
    ).reindex(columns=[m for m in _month_order if m in df_monthly["month"].values])

    if not pivot.empty:
        fig_cal = px.imshow(
            pivot,
            color_continuous_scale=[
                [0.0, "#ef4444"], [0.5, "#1e293b"], [1.0, "#22c55e"],
            ],
            color_continuous_midpoint=0,
            text_auto=".1f",
            aspect="auto",
        )
        fig_cal.update_layout(
            height=min(50 * len(pivot) + 80, 400),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            coloraxis_colorbar=dict(title="Rdt (%)"),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Mois",
            yaxis_title="Année",
        )
        st.plotly_chart(fig_cal, width="stretch")

st.markdown("---")

# ===========================================================================
# SECTION 5 — Tableau comparatif
# ===========================================================================
st.subheader("🏆 Comparaison des stratégies")

comp_df = build_strategy_comparison(pl_metrics)

if not comp_df.empty:
    def _highlight_sharpe(val):
        try:
            v = float(val)
            if v >= 1.5:
                return "background-color: rgba(34,197,94,0.3)"
            if v >= 0.8:
                return "background-color: rgba(245,158,11,0.2)"
            return "background-color: rgba(239,68,68,0.2)"
        except Exception:
            return ""

    styled = comp_df.style.applymap(_highlight_sharpe, subset=["Sharpe"])
    styled = styled.format({
        "Rdt total (%)":     "{:+.1f}%",
        "Rdt annualisé (%)": "{:+.2f}%",
        "B&H annualisé (%)": "{:+.2f}%",
        "Alpha (%)":         "{:+.2f}%",
        "Sharpe":            "{:.3f}",
        "Sortino":           "{:.3f}",
        "Max Drawdown (%)":  "{:.1f}%",
        "Calmar":            "{:.3f}",
        "Win Rate (%)":      "{:.1f}%",
        "Profit Factor":     "{:.2f}",
        "Expo (%)":          "{:.1f}%",
    }, na_rep="—")
    st.dataframe(styled, width="stretch")

# Contexte académique
st.markdown("---")
with st.expander("📚 Note méthodologique"):
    st.markdown("""
    - **Protocole walk-forward expanding** — aucun signal produit sur données vues à l'entraînement (zéro look-ahead bias).
    - **Stratégie simulée** : Long or si signal = **Haussier (+1)**. Neutre sinon (pas de short — cohérent avec achat physique).
    - **Frais** : coût aller-retour appliqué uniquement aux entrées/sorties de position (changement de signal).
    - **Sharpe** : rendement annualisé / volatilité des rendements quotidiens de la stratégie × √252. Taux sans risque = 0 (conservateur).
    - **Max Drawdown** : pire baisse depuis un sommet du capital.
    - **Alpha** : surperformance annualisée vs Buy & Hold (détenir de l'or en continu).
    - **Interprétation** : un Sharpe > 1 sur données OOS est un résultat solide. Les performances passées ne garantissent pas les performances futures.
    """)

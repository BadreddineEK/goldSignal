"""
04_portfolio.py — Simulateur d'Investissement GoldSignal

3 sections stateless (aucune BDD requise — fonctionne sur Cloud) :
  1. Simulateur historique  : « Si j'avais investi X€ le [date], j'aurais combien aujourd'hui ? »
  2. Timing d'achat         : « Est-ce le bon moment maintenant ? » (RSI, Bollinger, signal ML)
  3. Projection             : scénarios bull/base/bear + ML sur horizon libre
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import yfinance as yf


# ---------------------------------------------------------------------------
# Constantes visuelles (adaptatives clair/sombre)
# ---------------------------------------------------------------------------
_PAPER = _PLOT = "rgba(0,0,0,0)"
_GRID  = "rgba(128,128,128,0.18)"
_GOLD  = "#f59e0b"
_GREEN = "#22c55e"
_RED   = "#ef4444"
_BLUE  = "#60a5fa"
_GREY  = "#94a3b8"

# ---------------------------------------------------------------------------
# Définition des métaux
# ---------------------------------------------------------------------------
METALS = {
    "🥇 Or (XAU/EUR)": {
        "ticker_price": "GC=F",
        "unit_label": "g fin",
        "troy_oz": True,
        "currency": "USD",
        "name_short": "Or",
    },
    "🥈 Argent (XAG/EUR)": {
        "ticker_price": "SI=F",
        "unit_label": "g fin",
        "troy_oz": True,
        "currency": "USD",
        "name_short": "Argent",
    },
    "⚪ Platine (XPT/EUR)": {
        "ticker_price": "PL=F",
        "unit_label": "troy oz",
        "troy_oz": False,
        "currency": "USD",
        "name_short": "Platine",
    },
    "⚫ Palladium (XPD/EUR)": {
        "ticker_price": "PA=F",
        "unit_label": "troy oz",
        "troy_oz": False,
        "currency": "USD",
        "name_short": "Palladium",
    },
}

# ---------------------------------------------------------------------------
# Helpers de téléchargement (cachés)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _load_raw(ticker_price: str, start: str) -> tuple[pd.Series, pd.Series]:
    """Retourne (prix_usd, eurusd) alignés sur le même index journalier."""
    df_p  = yf.download(ticker_price, start=start, auto_adjust=True, progress=False)
    df_fx = yf.download("EURUSD=X",   start=start, auto_adjust=True, progress=False)
    price_usd = df_p["Close"].squeeze().rename("price_usd")
    eurusd    = df_fx["Close"].squeeze().rename("eurusd")
    both = pd.concat([price_usd, eurusd], axis=1).ffill().dropna()
    return both["price_usd"], both["eurusd"]


def _eur_per_gram(price_usd: pd.Series, eurusd: pd.Series, troy_oz: bool) -> pd.Series:
    """Convertit USD/troy-oz → EUR/g (ou EUR/troy-oz si troy_oz=False)."""
    eur_oz = price_usd / eurusd
    return (eur_oz / 31.1035) if troy_oz else eur_oz


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("💰 Simulateur d'Investissement")
st.caption(
    "Outil stateless — aucune donnée n'est stockée. "
    "Résultats basés sur les cours historiques yfinance."
)

# ---------------------------------------------------------------------------
# Sélecteur de métal global
# ---------------------------------------------------------------------------
metal_label = st.selectbox("Métal", list(METALS.keys()), key="sim_metal")
meta = METALS[metal_label]

start_load = (date.today() - timedelta(days=5 * 365)).isoformat()
with st.spinner("Chargement des cours…"):
    try:
        price_usd_s, eurusd_s = _load_raw(meta["ticker_price"], start_load)
        price_eur = _eur_per_gram(price_usd_s, eurusd_s, meta["troy_oz"])
        data_ok = len(price_eur) > 20
    except Exception as e:
        st.error(f"Impossible de charger les données : {e}")
        st.stop()

if not data_ok:
    st.warning("Données insuffisantes, réessaie dans quelques instants.")
    st.stop()

current_price = float(price_eur.iloc[-1])
current_date  = price_eur.index[-1].date()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric(f"Prix actuel ({meta['unit_label']})", f"{current_price:,.2f} €")
chg_1d  = (price_eur.iloc[-1] / price_eur.iloc[-2]   - 1) * 100 if len(price_eur) > 1   else 0
chg_30d = (price_eur.iloc[-1] / price_eur.iloc[-22]  - 1) * 100 if len(price_eur) > 22  else 0
chg_1y  = (price_eur.iloc[-1] / price_eur.iloc[-252] - 1) * 100 if len(price_eur) > 252 else 0
col_b.metric("Variation 1j",   f"{chg_1d:+.2f} %",  delta=f"{chg_1d:+.2f}%")
col_c.metric("Variation 30j",  f"{chg_30d:+.2f} %")
col_d.metric("Variation 1 an", f"{chg_1y:+.2f} %")

st.markdown("---")


# ===========================================================================
# SECTION 1 — Simulateur historique
# ===========================================================================
st.subheader("📅 Simulateur historique")
st.caption(
    "Calcule la performance d'un investissement passé jusqu'à aujourd'hui. "
    "Le calcul inclut uniquement la variation de cours — hors frais et spread achat/vente."
)

with st.expander("⚙️ Paramètres de simulation", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        min_date = price_eur.index[0].date()
        sim_date = st.date_input(
            "Date d'investissement",
            value=date.today() - timedelta(days=365),
            min_value=min_date,
            max_value=date.today() - timedelta(days=1),
            key="sim_date",
        )
    with c2:
        sim_mode = st.radio(
            "Saisir en…",
            ["Montant (€)", f"Unités ({meta['unit_label']})"],
            horizontal=True,
            key="sim_mode",
        )
    with c3:
        if sim_mode == "Montant (€)":
            sim_amount = st.number_input(
                "Montant investi (€)", min_value=1.0, value=1000.0, step=100.0, key="sim_amount"
            )
        else:
            sim_qty = st.number_input(
                f"Quantité ({meta['unit_label']})",
                min_value=0.001, value=1.0, step=0.1, key="sim_qty"
            )

if st.button("📊 Calculer la simulation", key="btn_sim", type="primary"):
    sim_ts    = pd.Timestamp(sim_date)
    idx       = price_eur.index.searchsorted(sim_ts)
    idx       = min(idx, len(price_eur) - 1)
    buy_price = float(price_eur.iloc[idx])
    actual_date = price_eur.index[idx].date()

    if sim_mode == "Montant (€)":
        invested = float(sim_amount)
        qty      = invested / buy_price
    else:
        qty      = float(sim_qty)
        invested = qty * buy_price

    current_value = qty * current_price
    gain_eur  = current_value - invested
    gain_pct  = gain_eur / invested * 100
    n_days    = (current_date - actual_date).days
    n_years   = n_days / 365.25
    cagr      = ((current_value / invested) ** (1 / n_years) - 1) * 100 if n_years > 0.01 else 0.0

    st.markdown("#### Résultats")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Investi",          f"{invested:,.2f} €")
    m2.metric("Valeur actuelle",  f"{current_value:,.2f} €")
    m3.metric("Gain / Perte",     f"{gain_eur:+,.2f} €", delta=f"{gain_pct:+.1f}%")
    m4.metric("Annualisé (CAGR)", f"{cagr:+.2f} %/an")
    m5.metric("Durée",            f"{n_days:,} j ({n_years:.1f} ans)")

    if gain_pct >= 0:
        st.success(f"✅ Bonne affaire ! Votre investissement a pris **{gain_pct:.1f}%** en {n_days} jours.")
    else:
        st.warning(f"📉 En moins-value de **{abs(gain_pct):.1f}%** sur la période.")

    # Courbe d'évolution
    slice_eur   = price_eur[price_eur.index >= price_eur.index[idx]]
    portf_value = slice_eur * qty

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=portf_value.index, y=portf_value.values,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(245,158,11,0.15)",
        line=dict(color=_GOLD, width=2),
        name="Valeur portefeuille",
        hovertemplate="%{x|%d %b %Y}<br>Valeur : %{y:,.2f} €<extra></extra>",
    ))
    fig_hist.add_hline(
        y=invested, line_dash="dot", line_color=_GREY,
        annotation_text=f"Investi : {invested:,.0f} €",
        annotation_position="top left",
    )
    fig_hist.update_layout(
        title=f"Évolution de la valeur depuis le {actual_date.strftime('%d/%m/%Y')}",
        xaxis_title=None,
        yaxis_title="Valeur (€)",
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
        hovermode="x unified",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Comparaison avec autres actifs
    st.markdown("#### Comparaison avec d'autres actifs sur la même période")
    bench_tickers = {
        "CAC 40":        "^FCHI",
        "S&P 500 (EUR)": "^GSPC",
        "Bitcoin (EUR)": "BTC-EUR",
    }
    bench_start = price_eur.index[idx].strftime("%Y-%m-%d")
    bench_rows  = []
    for bn, bt in bench_tickers.items():
        try:
            bdf = yf.download(bt, start=bench_start, auto_adjust=True, progress=False)
            if len(bdf) < 2:
                continue
            b_close = bdf["Close"].squeeze()
            b_ret   = (float(b_close.iloc[-1]) / float(b_close.iloc[0]) - 1) * 100
            bench_rows.append({"Actif": bn, "Performance": b_ret})
        except Exception:
            pass
    bench_rows.append({"Actif": meta["name_short"], "Performance": gain_pct})

    bench_df = pd.DataFrame(bench_rows).sort_values("Performance", ascending=False)
    colors   = [
        _GOLD if r["Actif"] == meta["name_short"]
        else (_GREEN if r["Performance"] >= 0 else _RED)
        for _, r in bench_df.iterrows()
    ]

    fig_bench = go.Figure(go.Bar(
        x=bench_df["Actif"], y=bench_df["Performance"],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in bench_df["Performance"]],
        textposition="outside",
    ))
    fig_bench.update_layout(
        title="Performance comparée (même période)",
        yaxis_title="Rendement total (%)",
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        yaxis=dict(gridcolor=_GRID),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_bench, use_container_width=True)

st.markdown("---")

# ===========================================================================
# SECTION 2 — Timing d'achat
# ===========================================================================
st.subheader("🎯 Est-ce le bon moment pour acheter ?")
st.caption(
    "Analyse technique rapide pour évaluer si le cours est dans une zone d'opportunité. "
    "Ce n'est pas un conseil financier — outil d'aide à la réflexion uniquement."
)

n_tech = min(252, len(price_eur))
p_tech = price_eur.iloc[-n_tech:].copy()

rsi_14   = float(_rsi(p_tech).iloc[-1])
sma20    = float(p_tech.rolling(20).mean().iloc[-1])
sma50    = float(p_tech.rolling(50).mean().iloc[-1]) if n_tech >= 50  else float("nan")
sma200   = float(p_tech.rolling(200).mean().iloc[-1]) if n_tech >= 200 else float("nan")
bb_mid   = float(p_tech.rolling(20).mean().iloc[-1])
bb_std   = float(p_tech.rolling(20).std().iloc[-1])
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_pct   = (current_price - bb_lower) / (bb_upper - bb_lower) * 100

pct_rank_1y = float((p_tech.iloc[-252:] < current_price).mean() * 100) if n_tech >= 252 else 50.0

score_rsi   = max(0, min(100, (70 - rsi_14)         / 40 * 100))
score_bb    = max(0, min(100, (50 - bb_pct)          / 50 * 100))
score_pct   = max(0, min(100, (50 - pct_rank_1y)     / 50 * 100))
score_trend = 70.0 if (not np.isnan(sma200) and current_price > sma200) else 30.0

opp_score = 0.30 * score_rsi + 0.25 * score_bb + 0.25 * score_pct + 0.20 * score_trend

ml_signal_txt = None
_raw_sig_t = st.session_state.get("latest_signal")
if _raw_sig_t is not None:
    _s = _raw_sig_t["signal"] if isinstance(_raw_sig_t, dict) else _raw_sig_t
    try:
        sig = int(_s)
        ml_signal_txt = {1: "🟢 ML prédit une hausse", 0: "⚪ ML prédit une stabilité", -1: "🔴 ML prédit une baisse"}.get(sig)
        opp_score = opp_score * 0.6 + (70 if sig == 1 else 20 if sig == -1 else 50) * 0.4
    except (TypeError, ValueError):
        pass

col_score, col_indicators = st.columns([1, 2])

with col_score:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(opp_score, 1),
        number={"suffix": "/100", "font": {"size": 28}},
        title={"text": "Score opportunité achat", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": _GOLD if opp_score >= 60 else (_GREEN if opp_score >= 40 else _RED)},
            "steps": [
                {"range": [0,  35], "color": "rgba(239,68,68,0.15)"},
                {"range": [35, 65], "color": "rgba(148,163,184,0.15)"},
                {"range": [65, 100], "color": "rgba(34,197,94,0.15)"},
            ],
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor=_PAPER,
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if opp_score >= 65:
        st.success("🟢 **Zone d'achat** — prix relativement bas, momentum favorable")
    elif opp_score >= 40:
        st.info("⚪ **Zone neutre** — pas de signal fort dans un sens ou l'autre")
    else:
        st.warning("🔴 **Zone de prudence** — prix élevé historiquement")

    if ml_signal_txt:
        st.caption(f"Signal ML : {ml_signal_txt}")

    # Détail du score
    with st.expander("📊 Détail du score opportunité"):
        st.markdown("""
        Le **score d'opportunité** (0–100) est une combinaison pondérée de 4 composantes :

        | Composante | Poids | Logique |
        |---|---|---|
        | RSI 14j | 30% | Bas RSI = survente = achat potentiel |
        | Position Bollinger | 25% | Bas de bande = prix bas historiquement |
        | Percentile 1 an | 25% | < 25e centile = prix bas sur 1 an |
        | Tendance SMA 200 | 20% | Au-dessus = tendance positive |

        Si un **signal ML** est disponible (page Prédictions), il remplace 40% du score
        technique (60% compos. tech. + 40% signal ML).

         
        **Interprétation** :
        - **65–100** 🟢 : zone d’achat — confluence de signaux favorables
        - **35–65** ⚪ : zone neutre — pas de signal fort
        - **0–35** 🔴 : zone de prudence — prix élevé ou suracheté

        ⚠️ *Ce score est un outil de réflexion, pas une recommandation d’achat.*
        """)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("RSI (30%)",         f"{score_rsi:.0f}/100",  f"{0.30*score_rsi:.1f} pts")
        sc2.metric("Bollinger (25%)",   f"{score_bb:.0f}/100",   f"{0.25*score_bb:.1f} pts")
        sc3.metric("Percentile (25%)",  f"{score_pct:.0f}/100",  f"{0.25*score_pct:.1f} pts")
        sc4.metric("SMA 200 (20%)",     f"{score_trend:.0f}/100",f"{0.20*score_trend:.1f} pts")

with col_indicators:
    st.markdown("##### Indicateurs détaillés")
    inds = [
        ("RSI 14j",            f"{rsi_14:.1f}",
         "< 30 = survente | 30–70 = normal | > 70 = surachat",
         _GREEN if rsi_14 < 40 else (_RED if rsi_14 > 60 else _GREY)),
        ("Position Bollinger (BB%)", f"{bb_pct:.0f}%",
         "0% = bas bande inf. (opportunité) | 100% = bande sup. (méfiance)",
         _GREEN if bb_pct < 25 else (_RED if bb_pct > 75 else _GREY)),
        ("Percentile 1 an",    f"{pct_rank_1y:.0f}e",
         "Ex : 20e = prix inférieur à 80% des jours de l’année → prix bas",
         _GREEN if pct_rank_1y < 30 else (_RED if pct_rank_1y > 70 else _GREY)),
        ("vs SMA 200j",        f"{current_price / sma200 * 100:.1f}%" if not np.isnan(sma200) else "N/A",
         "Moyenne mobile 200j. > 100% = tendance haussière. Signal long-terme.",
         _GREEN if (not np.isnan(sma200) and current_price > sma200) else _RED),
        ("vs SMA 50j",         f"{current_price / sma50 * 100:.1f}%" if not np.isnan(sma50) else "N/A",
         "Moyenne mobile 50j. > 100% = momentum positif court terme.",
         _GREEN if (not np.isnan(sma50) and current_price > sma50) else _RED),
    ]
    for label, val, help_txt, color in inds:
        ic1, ic2, ic3 = st.columns([2, 1, 3])
        ic1.markdown(f"**{label}**")
        ic2.markdown(f'<span style="color:{color};font-weight:700">{val}</span>', unsafe_allow_html=True)
        ic3.caption(help_txt)

    n_bb = min(90, len(p_tech))
    p_bb = p_tech.iloc[-n_bb:]
    bb_m = p_bb.rolling(20).mean()
    bb_s = p_bb.rolling(20).std()

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=p_bb.index, y=(bb_m + 2 * bb_s).values,
        mode="lines", line=dict(color=_RED, width=1, dash="dot"),
        name="BB sup",
    ))
    fig_bb.add_trace(go.Scatter(
        x=p_bb.index, y=(bb_m - 2 * bb_s).values,
        mode="lines", line=dict(color=_GREEN, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
        name="BB inf",
    ))
    fig_bb.add_trace(go.Scatter(
        x=p_bb.index, y=p_bb.values,
        mode="lines", line=dict(color=_GOLD, width=1.8),
        name=meta["name_short"],
    ))
    fig_bb.update_layout(
        title="Prix + Bandes de Bollinger (90 derniers jours)",
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID, title=f"€/{meta['unit_label']}"),
        height=220, margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="h", y=1.15, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_bb, use_container_width=True)

st.markdown("---")

# ===========================================================================
# SECTION 3 — Projection Monte-Carlo
# ===========================================================================
st.subheader("🔮 Projection Monte-Carlo : que se passe-t-il si j’investis maintenant ?")
st.caption(
    "Scénarios basés sur la **distribution historique des rendements** (2 ans). "
    "Bull = 75e centile, Base = médiane, Bear = 25e centile. "
    "❗ Les performances passées ne garantissent pas les performances futures."
)

with st.expander("📊 Méthodologie Monte-Carlo — comment fonctionne la simulation ?"):
    st.markdown("""
    La simulation utilise un **mouvement brownien géométrique** (GBM) : hypothèse standard
    en finance pour modéliser les prix d’actifs.

    **Étapes** :
    1. Calcul des log-rendements journaliers sur les 2 dernières années : $r_t = \\ln(P_t / P_{t-1})$
    2. Estimation de la **dérive** $\\mu$ (moyenne) et la **volatilité** $\\sigma$ (écart-type)
    3. Génération de $N$ chemins aléatoires de longueur $H$ jours :
       $P_{t+1} = P_t \\times e^{r_t}$ avec $r_t \\sim \\mathcal{N}(\\mu, \\sigma)$
    4. Lecture des percentiles 5/25/50/75/95 à l’horizon choisi

    **Limites du modèle** :
    - Supporte des log-rendements gaussiens *i.i.d.* — sous-estime les queues épaisses (événements extrêmes)
    - Ne modélise pas les régimes de marché (crise, vol cluster)
    - $\\mu$ et $\\sigma$ calibrés sur l’historique récent ne sont pas stables dans le temps

    **Interprétation** :
    - **Intervalle de confiance 90%** (P5–P95) : 9 simulations sur 10 tombent dans cette zone
    - **P50 (Base)** : scénario médian, autant de chances de finir au-dessus qu’en dessous
    - **Probabilité de perte** : % de chemins se terminant en dessous du montant investi
    """)

with st.expander("⚙️ Paramètres de projection", expanded=True):
    p1, p2, p3 = st.columns(3)
    with p1:
        proj_amount  = st.number_input("Montant à investir (€)", min_value=1.0, value=1000.0, step=100.0, key="proj_amount")
    with p2:
        proj_horizon = st.slider("Horizon (jours)", min_value=5, max_value=365, value=30, step=5, key="proj_horizon")
    with p3:
        proj_sims    = st.number_input("Simulations Monte-Carlo", min_value=200, max_value=2000, value=500, step=100, key="proj_sims")

if st.button("🔮 Lancer la projection", key="btn_proj", type="primary"):
    n_hist  = min(504, len(price_eur) - 1)
    log_ret = np.log(price_eur / price_eur.shift(1)).dropna().iloc[-n_hist:].values
    mu      = float(np.mean(log_ret))
    sigma   = float(np.std(log_ret))

    rng    = np.random.default_rng(42)
    shocks = rng.normal(mu, sigma, size=(int(proj_sims), proj_horizon))
    paths  = proj_amount * np.exp(np.cumsum(shocks, axis=1))

    final_vals = paths[:, -1]
    p25  = float(np.percentile(final_vals, 25))
    p50  = float(np.percentile(final_vals, 50))
    p75  = float(np.percentile(final_vals, 75))
    p05  = float(np.percentile(final_vals, 5))
    p95  = float(np.percentile(final_vals, 95))
    prob_loss = float((final_vals < proj_amount).mean() * 100)

    st.markdown("#### Scénarios à l'horizon")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🐻 Bear (P25)", f"{p25:,.2f} €", delta=f"{(p25/proj_amount-1)*100:+.1f}%")
    sc2.metric("📊 Base (P50)", f"{p50:,.2f} €", delta=f"{(p50/proj_amount-1)*100:+.1f}%")
    sc3.metric("🐂 Bull (P75)", f"{p75:,.2f} €", delta=f"{(p75/proj_amount-1)*100:+.1f}%")

    st.caption(
        f"Probabilité de perte : **{prob_loss:.0f}%** — "
        f"IC 90% : {p05:,.0f} € → {p95:,.0f} €  "
        f"(σ = {sigma*100:.2f}%/j calibré sur {n_hist} jours)"
    )

    _raw_sig_p = st.session_state.get("latest_signal")
    if _raw_sig_p is not None:
        _sp = _raw_sig_p["signal"] if isinstance(_raw_sig_p, dict) else _raw_sig_p
        try:
            _sig_int = int(_sp)
            ml_txt = {1: "🟢 Signal ML haussier — scénario Bull plus probable", 0: "⚪ Signal ML neutre", -1: "🔴 Signal ML baissier — scénario Bear à surveiller"}.get(_sig_int)
            if ml_txt:
                st.info(f"**Signal ML (page Prédictions) :** {ml_txt}")
        except (TypeError, ValueError):
            pass

    # Fan chart
    x_days   = list(range(1, proj_horizon + 1))
    p25_path = np.percentile(paths, 25, axis=0)
    p50_path = np.percentile(paths, 50, axis=0)
    p75_path = np.percentile(paths, 75, axis=0)
    p05_path = np.percentile(paths,  5, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=x_days + x_days[::-1],
        y=list(p95_path) + list(p05_path[::-1]),
        fill="toself", fillcolor="rgba(96,165,250,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 90% (P5-P95)",
    ))
    fig_proj.add_trace(go.Scatter(
        x=x_days + x_days[::-1],
        y=list(p75_path) + list(p25_path[::-1]),
        fill="toself", fillcolor="rgba(245,158,11,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zone Bull/Bear (P25-P75)",
    ))
    fig_proj.add_trace(go.Scatter(x=x_days, y=p25_path, mode="lines", line=dict(color=_RED,   width=1.5, dash="dash"), name="Bear P25"))
    fig_proj.add_trace(go.Scatter(x=x_days, y=p50_path, mode="lines", line=dict(color=_GOLD,  width=2.5),              name="Base P50"))
    fig_proj.add_trace(go.Scatter(x=x_days, y=p75_path, mode="lines", line=dict(color=_GREEN, width=1.5, dash="dash"), name="Bull P75"))
    fig_proj.add_hline(
        y=proj_amount, line_dash="dot", line_color=_GREY,
        annotation_text=f"Investi : {proj_amount:,.0f} €",
        annotation_position="top left",
    )
    fig_proj.update_layout(
        title=f"Projection Monte-Carlo sur {proj_horizon} jours ({int(proj_sims):,} simulations)",
        xaxis_title="Jours", yaxis_title="Valeur (€)",
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # Distribution finale
    fig_dist = go.Figure(go.Histogram(
        x=final_vals, nbinsx=60,
        marker_color=_GOLD, opacity=0.75,
    ))
    fig_dist.add_vline(x=proj_amount, line_dash="dot",  line_color=_RED,   annotation_text="Investi")
    fig_dist.add_vline(x=p50,         line_dash="dash", line_color=_GREEN, annotation_text=f"Médiane {p50:,.0f}€")
    fig_dist.update_layout(
        title=f"Distribution des valeurs finales à J+{proj_horizon}",
        xaxis_title="Valeur (€)", yaxis_title="Fréquence",
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.caption(
        "ℹ️ La simulation suppose des log-rendements gaussiens i.i.d. calibrés sur l'historique récent. "
        "Elle ne tient pas compte des queues épaisses, des crises ou des chocs géopolitiques."
    )


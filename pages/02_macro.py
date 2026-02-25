"""
02_macro.py — Dashboard Macro & Technique GoldSignal (Module 2).

Sections :
  1. Score macro composite (5 axes) avec verdict
  2. Cours XAU/EUR et XAG/EUR avec SMA20/50/200 + Bollinger + RSI + MACD
  3. Ratio Or/Argent historique + percentile
  4. Corrélations rolling 60j (or vs DXY, taux réels, VIX)
  5. Heatmap corrélations mensuelles
  6. Saisonnalité mensuelle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.fetcher import (
    fetch_ticker, get_taux_reels, get_spot_xau_eur, get_spot_xag_eur,
)
from data.database import get_config
from data.processor import (
    ohlcv_usd_oz_to_eur_g, compute_gold_silver_ratio, compute_log_returns,
)
from analysis.technical import add_all_indicators, rsi as calc_rsi, atr as calc_atr
from analysis.correlations import (
    rolling_correlation, build_monthly_correlation_heatmap, taux_reels_vs_or,
)
from analysis.macro_score import compute_macro_score
from analysis.seasonality import seasonality_by_month
from utils.formatting import fmt_pct, fmt_ratio, verdict_emoji

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("📊 Macro & Technique")
st.caption("Contexte de marché, indicateurs techniques et corrélations.")

cfg_macro = get_config("macro") or {}
tickers = get_config("tickers_yfinance") or {}

# ---------------------------------------------------------------------------
# Chargement des données (mise en cache Streamlit 1h)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Chargement des données marché…")
def load_data() -> dict:
    """Charge et prépare toutes les séries nécessaires."""
    df_xau_usd = fetch_ticker(tickers.get("xau", "GC=F"))
    df_xag_usd = fetch_ticker(tickers.get("xag", "SI=F"))
    df_eurusd = fetch_ticker(tickers.get("eurusd", "EURUSD=X"))
    df_dxy = fetch_ticker(tickers.get("dxy", "DX-Y.NYB"))
    df_vix = fetch_ticker(tickers.get("vix", "^VIX"))
    df_sp500 = fetch_ticker(tickers.get("sp500", "^GSPC"))
    df_tnx = fetch_ticker(tickers.get("tnx", "^TNX"))
    s_taux_reels = get_taux_reels()

    df_xau_eur = ohlcv_usd_oz_to_eur_g(df_xau_usd, df_eurusd)
    df_xag_eur = ohlcv_usd_oz_to_eur_g(df_xag_usd, df_eurusd)

    # Indicateurs techniques sur or EUR
    df_xau_tech = add_all_indicators(df_xau_eur) if not df_xau_eur.empty else df_xau_eur
    df_xag_tech = add_all_indicators(df_xag_eur) if not df_xag_eur.empty else df_xag_eur

    return {
        "xau_eur": df_xau_tech,
        "xag_eur": df_xag_tech,
        "xau_usd": df_xau_usd,
        "xag_usd": df_xag_usd,
        "eurusd": df_eurusd,
        "dxy": df_dxy,
        "vix": df_vix,
        "sp500": df_sp500,
        "tnx": df_tnx,
        "taux_reels": s_taux_reels,
    }


data = load_data()

xau = data["xau_eur"]
xag = data["xag_eur"]
dxy = data["dxy"]
vix = data["vix"]
taux_reels = data["taux_reels"]

if xau.empty:
    st.error("Données de marché indisponibles. Vérifiez votre connexion internet.")
    st.stop()

# ---------------------------------------------------------------------------
# 1. Score Macro Composite
# ---------------------------------------------------------------------------
st.subheader("🎯 Score Macro Composite")

try:
    taux_reel_now = float(taux_reels.iloc[-1]) if not taux_reels.empty else 1.0
    vix_now = float(vix["close"].iloc[-1]) if not vix.empty else 20.0
    rsi_now = float(xau["rsi_14"].iloc[-1]) if "rsi_14" in xau.columns else 50.0

    ratio_series = compute_gold_silver_ratio(data["xau_usd"], data["xag_usd"] if "xag_usd" in data else pd.DataFrame())
    # ratio en prices (USD/oz) est plus standard
    xau_close = data["xau_usd"]["close"] if not data["xau_usd"].empty else xau["close"]
    xag_close_raw = data["xag_usd"]["close"] if not data["xag_usd"].empty else None
    if xag_close_raw is not None and not xag_close_raw.empty:
        ratio_series = xau_close / xag_close_raw.reindex(xau_close.index, method="ffill")
        ratio_now = float(ratio_series.dropna().iloc[-1])
    else:
        ratio_series = pd.Series(dtype=float)
        ratio_now = 80.0

    macro = compute_macro_score(
        taux_reel_dernier=taux_reel_now,
        close_dxy=dxy["close"] if not dxy.empty else pd.Series(dtype=float),
        vix_level=vix_now,
        rsi_or=rsi_now,
        ratio_or_argent=ratio_now,
        ratio_series=ratio_series,
        cfg=cfg_macro,
    )

    score = macro["score_total"]
    verdict = macro["verdict"]
    details = macro["details"]

    # Affichage score
    col_score, col_details = st.columns([1, 2])
    with col_score:
        color = {"favorable": "#22c55e", "neutre": "#f59e0b", "défavorable": "#ef4444"}[verdict]
        emoji = verdict_emoji(verdict)
        st.markdown(
            f"""
            <div style='text-align:center; padding:24px; background:{color}20;
                        border:2px solid {color}; border-radius:12px'>
              <div style='font-size:3em'>{emoji}</div>
              <div style='font-size:2.5em; font-weight:800; color:{color}'>{score:+.1f}/5</div>
              <div style='font-size:1.2em; color:{color}; font-weight:600'>{verdict.title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        axes = {
            "Taux réels": details["taux_reels"],
            "DXY tendance": details["dxy_trend"],
            "VIX niveau": details["vix_level"],
            "RSI or": details["rsi_or"],
            "Ratio Or/Ag": details["ratio_or_argent"],
        }
        df_axes = pd.DataFrame({
            "Axe": list(axes.keys()),
            "Score": list(axes.values()),
        })
        fig_radar = go.Figure(go.Bar(
            x=df_axes["Axe"],
            y=df_axes["Score"],
            marker_color=["#22c55e" if v > 0 else "#ef4444" if v < 0 else "#f59e0b"
                          for v in df_axes["Score"]],
            text=[f"{v:+.2f}" for v in df_axes["Score"]],
            textposition="auto",
        ))
        fig_radar.update_layout(
            yaxis=dict(
                range=[-1.5, 1.5],
                zeroline=True,
                zerolinecolor="rgba(128,128,128,0.5)",
                gridcolor="rgba(128,128,128,0.15)",
                tickformat="+.1f",
                title="Score (-1 défavorable → +1 favorable)",
            ),
            xaxis_title="",
            height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig_radar, width="stretch")
        st.caption(
            "▲ Score positif = contexte favorable à l'or | "
            "▼ Score négatif = vent contraire | "
            "Total : somme des 5 axes (−5 → +5)"
        )

    st.caption(
        f"Taux réels : {taux_reel_now:.2f}% | VIX : {vix_now:.1f} | "
        f"RSI or : {rsi_now:.1f} | Ratio or/argent : {ratio_now:.1f}"
    )

except Exception as exc:
    st.warning(f"Score macro non calculable : {exc}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Cours XAU/EUR avec indicateurs techniques
# ---------------------------------------------------------------------------
st.subheader("📈 Cours Or (€/g fin) — Analyse Technique")
st.caption(
    "**Bandes de Bollinger** : canal de volatilité (2σ). Prix proche de la bande supérieure = "
    "sur-acheté, proche de l'inférieure = survente. "
    "**SMA** : moyennes mobiles simples 20j/50j/200j. "
    "**RSI** : force relative 0-100 (>70 sur-acheté, <30 survente). "
    "**MACD** : momentum (croisement ligne/signal = signal directionnel)."
)

# Sélecteur de période
period_opts = {"6 mois": 180, "1 an": 365, "3 ans": 1095, "5 ans": 1825, "Tout": 0}
period_label = st.selectbox("Période", list(period_opts.keys()), index=1)
n_days = period_opts[period_label]
df_plot = xau.iloc[-n_days:] if n_days > 0 else xau

_GRID = "rgba(128,128,128,0.18)"
_PAPER = "rgba(0,0,0,0)"
_PLOT  = "rgba(0,0,0,0)"

# Graphe prix + SMA + Bollinger
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.06,
    subplot_titles=[
        "💰 Prix Or (€/g fin) + Bandes de Bollinger + Moyennes mobiles",
        "📊 RSI 14 jours (Relative Strength Index)",
        "➡️ MACD (momentum directionnel)",
    ],
)

# Chandelier
if "open" in df_plot.columns:
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"],
        name="XAU/EUR €/g",
        increasing_line_color="#22c55e", increasing_fillcolor="rgba(34,197,94,0.7)",
        decreasing_line_color="#ef4444", decreasing_fillcolor="rgba(239,68,68,0.7)",
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["close"], mode="lines",
        name="XAU/EUR €/g", line=dict(color="#f59e0b", width=2),
    ), row=1, col=1)

# Bollinger
if "bb_upper" in df_plot.columns:
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["bb_upper"],
        mode="lines", name="Bollinger sup (sur-acheté)",
        line=dict(color="#818cf8", dash="dot", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["bb_lower"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(129,140,248,0.08)",
        name="Bollinger inf (survente)",
        line=dict(color="#818cf8", dash="dot", width=1.2),
    ), row=1, col=1)

# SMA
for col_name, color, label in [
    ("sma20",  "#38bdf8", "SMA 20j"),
    ("sma50",  "#f472b6", "SMA 50j"),
    ("sma200", "#fb923c", "SMA 200j"),
]:
    if col_name in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col_name], mode="lines",
            name=label, line=dict(color=color, width=1.3),
        ), row=1, col=1)

# RSI avec zones colorées
if "rsi_14" in df_plot.columns:
    # Zone survente (verte) et sur-achat (rouge)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,0.08)",  line_width=0, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["rsi_14"], mode="lines",
        name="RSI 14j", line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)",
    ), row=2, col=1)
    for y, color, label in [
        (70, "rgba(239,68,68,0.7)",  "Sur-acheté (70)"),
        (50, "rgba(148,163,184,0.5)", "Neutre (50)"),
        (30, "rgba(34,197,94,0.7)",  "Survente (30)"),
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10, row=2, col=1)

# MACD
if "macd_line" in df_plot.columns:
    hist_vals = df_plot["macd_hist"].fillna(0)
    fig.add_trace(go.Bar(
        x=df_plot.index, y=hist_vals,
        name="Histogramme MACD",
        marker_color=["rgba(34,197,94,0.6)" if v >= 0 else "rgba(239,68,68,0.6)" for v in hist_vals],
        showlegend=True,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["macd_line"], mode="lines",
        name="MACD", line=dict(color="#34d399", width=1.5),
    ), row=3, col=1)
    if "macd_signal" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["macd_signal"], mode="lines",
            name="Signal", line=dict(color="#fb923c", width=1.5),
        ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.4)", row=3, col=1)

fig.update_layout(
    height=680,
    paper_bgcolor=_PAPER,
    plot_bgcolor=_PLOT,
    legend=dict(orientation="h", y=1.03, x=0, font=dict(size=11)),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=0, r=10, t=50, b=0),
)
fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
fig.update_xaxes(gridcolor=_GRID)
fig.update_yaxes(title_text="€/g", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)
st.plotly_chart(fig, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# 3. Ratio Or/Argent
# ---------------------------------------------------------------------------
st.subheader("⚖️ Ratio Or/Argent")

if not data["xau_usd"].empty:
    try:
        xag_usd_raw = data["xag_usd"]
        ratio_full = data["xau_usd"]["close"] / xag_usd_raw["close"].reindex(
            data["xau_usd"].index, method="ffill"
        )
        ratio_plot = ratio_full.iloc[-n_days:] if n_days > 0 else ratio_full
        p10 = float(ratio_full.quantile(0.10))
        p50 = float(ratio_full.quantile(0.50))
        p90 = float(ratio_full.quantile(0.90))
        current_ratio = float(ratio_full.dropna().iloc[-1])
        pct_rank = float((ratio_full < current_ratio).mean() * 100)

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=ratio_plot.index, y=ratio_plot.values,
            mode="lines", name="Ratio", line=dict(color="#f59e0b", width=1.5),
        ))
        for level, label, color in [(p10, "P10", "#22c55e"), (p50, "P50", "#94a3b8"), (p90, "P90", "#ef4444")]:
            fig_ratio.add_hline(
                y=level, line_dash="dot", line_color=color,
                annotation_text=f"{label}: {level:.1f}",
                annotation_position="right",
            )

        fig_ratio.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.18)"),
            margin=dict(l=0, r=60, t=10, b=0),
        )
        st.plotly_chart(fig_ratio, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ratio actuel", f"{current_ratio:.1f}")
        c2.metric("Percentile historique", f"{pct_rank:.0f}e")
        c3.metric("P10 historique", f"{p10:.1f}")
        c4.metric("P90 historique", f"{p90:.1f}")

        if pct_rank >= 75:
            st.info("🪙 Ratio élevé : l'argent est historiquement bon marché relatif à l'or.")
        elif pct_rank <= 25:
            st.info("🥇 Ratio bas : l'or est historiquement bon marché relatif à l'argent.")
    except Exception as exc:
        st.warning(f"Ratio non calculable : {exc}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 4. Corrélations Rolling
# ---------------------------------------------------------------------------
st.subheader("🔗 Corrélations Rolling 60 jours")
st.caption(
    "Une corrélation de **-1** = mouvements exactement inverses. **+1** = mouvements identiques. **0** = indépendants.\n"
    "Théoriquement : **or vs DXY** tendanciellement négatif (dollar fort → or baisse), "
    "**or vs taux réels** négatif (taux hausse → coût d'opportunité de détenir de l'or → baisse). "
    "Ces relations ne tiennent pas toujours — la corrélation glissante permet de surveiller leur stabilité."
)

try:
    log_xau = compute_log_returns(xau["close"], [1])["log_return_1"].dropna()
    corr_data = {}

    if not dxy.empty:
        log_dxy = compute_log_returns(dxy["close"], [1])["log_return_1"].dropna()
        corr_data["Or vs DXY (dollar)"] = rolling_correlation(log_xau, log_dxy, 60)

    if not vix.empty:
        log_vix = compute_log_returns(vix["close"], [1])["log_return_1"].dropna()
        corr_data["Or vs VIX (volatilité)"] = rolling_correlation(log_xau, log_vix, 60)

    if not taux_reels.empty:
        diff_taux = taux_reels.diff().dropna()
        corr_data["Or vs Taux réels TIPS"] = rolling_correlation(log_xau, diff_taux, 60)

    if corr_data:
        fig_corr = go.Figure()

        # Zone de corrélation forte
        fig_corr.add_hrect(y0=0.5,  y1=1,  fillcolor="rgba(34,197,94,0.05)",  line_width=0)
        fig_corr.add_hrect(y0=-1,   y1=-0.5, fillcolor="rgba(239,68,68,0.05)", line_width=0)

        colors_corr = ["#38bdf8", "#f472b6", "#34d399"]
        for i, (name, series) in enumerate(corr_data.items()):
            s = series.iloc[-n_days:] if n_days > 0 else series
            fig_corr.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=name,
                line=dict(color=colors_corr[i % len(colors_corr)], width=2),
            ))

        fig_corr.add_hline(y=0, line_color="rgba(148,163,184,0.6)", line_dash="dash",
                           annotation_text="Indépendant (0)", annotation_position="left")
        fig_corr.add_hline(y=0.5,  line_color="rgba(34,197,94,0.5)",  line_dash="dot",
                           annotation_text="Corr. forte (+0.5)",  annotation_position="right", annotation_font_size=10)
        fig_corr.add_hline(y=-0.5, line_color="rgba(239,68,68,0.5)", line_dash="dot",
                           annotation_text="Corr. inverse (-0.5)", annotation_position="right", annotation_font_size=10)

        fig_corr.update_layout(
            height=320,
            yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID,
                       title="Corrélation (Pearson, rendements 1j)",
                       tickvals=[-1, -0.5, 0, 0.5, 1],
                       ticktext=["-1\n(inverse)", "-0.5", "0\n(indép.)", "+0.5", "+1\n(identique)"]),
            xaxis=dict(gridcolor=_GRID),
            paper_bgcolor=_PAPER,
            plot_bgcolor=_PLOT,
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=0, r=120, t=10, b=0),
        )
        st.plotly_chart(fig_corr, width="stretch")
except Exception as exc:
    st.warning(f"Corrélations non disponibles : {exc}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 5. Heatmap corrélations mensuelles
# ---------------------------------------------------------------------------
st.subheader("🌡️ Heatmap Corrélations Mensuelles")
st.caption(
    "Corrélation de Pearson entre les rendements mensuels de l'or et des actifs macro. "
    "**Bleu = corrélation positive** (montent ensemble), **Rouge = inverse** (montent en sens opposé). "
    "Utile pour comprendre la diversification de l'or dans un portefeuille."
)

try:
    prices_dict = {"Or (€)": xau["close"], "DXY": dxy["close"]}
    if not vix.empty:
        prices_dict["VIX"] = vix["close"]
    if not taux_reels.empty:
        prices_dict["Taux réels"] = taux_reels

    corr_matrix = build_monthly_correlation_heatmap(prices_dict)
    fig_heat = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        labels=dict(color="Corrélation"),
    )
    fig_heat.update_traces(textfont_size=13)
    fig_heat.update_layout(
        height=320,
        paper_bgcolor=_PAPER,
        coloraxis_colorbar=dict(
            title="Corrélation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1 inverse", "-0.5", "0 indép.", "+0.5", "+1 parallèle"],
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_heat, width="stretch")
except Exception as exc:
    st.warning(f"Heatmap non disponible : {exc}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 6. Saisonnalité mensuelle
# ---------------------------------------------------------------------------
st.subheader("📅 Saisonnalité Mensuelle (depuis 2000)")
st.caption(
    "Rendement moyen de l'or (€/g) par mois de l'année, basé sur plus de 20 ans de données. "
    "Les barres d'erreur représentent l'écart-type (variabilité historique). "
    "**Attention** : la saisonnalité est un biais structurel moyen — pas une prédiction certaine."
)

try:
    stats_saison = seasonality_by_month(xau["close"])
    vals = stats_saison["rendement_moyen_pct"]
    fig_saison = go.Figure()

    # Zone zéro
    fig_saison.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148,163,184,0.5)",
        annotation_text="0% (neutre)",
        annotation_position="left",
    )

    fig_saison.add_trace(go.Bar(
        x=stats_saison["mois_nom"],
        y=vals,
        error_y=dict(
            type="data",
            array=stats_saison["rendement_std_pct"].values,
            visible=True,
            color="rgba(148,163,184,0.6)",
        ),
        marker_color=["rgba(34,197,94,0.75)" if v > 0 else "rgba(239,68,68,0.75)" for v in vals],
        marker_line_color=["#22c55e" if v > 0 else "#ef4444" for v in vals],
        marker_line_width=1.5,
        text=vals.apply(lambda v: f"{v:+.2f}%"),
        textposition="outside",
        textfont=dict(size=11),
        name="Rendement moyen",
    ))

    fig_saison.update_layout(
        yaxis_title="Rendement moyen mensuel (%)",
        yaxis=dict(gridcolor=_GRID, zeroline=True, zerolinecolor="rgba(148,163,184,0.5)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=360,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_saison, width="stretch")
    st.caption(
        "Barres d'erreur = écart-type historique. "
        "Basé sur les clôtures mensuelles XAU/EUR (€/g fin) depuis 2000."
    )
except Exception as exc:
    st.warning(f"Saisonnalité non disponible : {exc}")

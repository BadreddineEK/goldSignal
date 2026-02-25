"""
06_benchmark.py — Tableau de bord de comparaison des modèles (vitrine ML).

Lit les résultats entraînés en page 03 (st.session_state) et affiche :
  1. Tableau comparatif multi-métriques (DA%, Brier, Log-Loss, RMSE)
  2. Radar chart multi-critères (vue synthétique)
  3. Courbes d'apprentissage LSTM (convergence train/val par fold)
  4. Heatmap des poids d'attention LSTM (interprétabilité)
  5. Matrices de confusion normalisées (tous modèles)
  6. Directional Accuracy par fold — stabilité temporelle
  7. Courbes de calibration (fiabilité probabiliste)
  8. Test de Diebold-Mariano — significativité statistique
  9. Poids du méta-apprenant hybride (contribution de chaque modèle)
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

st.header("📐 Benchmark des Modèles ML")
st.caption(
    "Évaluation rigoureuse en walk-forward — tous les résultats proviennent "
    "d'un protocole zéro-leakage. Lancez d'abord l'entraînement sur la page **Prédictions ML**."
)

# ---------------------------------------------------------------------------
# Vérification des données disponibles
# ---------------------------------------------------------------------------
KEYS = ["ml_results", "arima_result", "lstm_result", "hybrid_result", "rw_metrics"]
available = {k: st.session_state.get(k) for k in KEYS}

if not available["ml_results"]:
    st.warning("⚠️ Aucun modèle entraîné. Rendez-vous sur **🤖 Prédictions ML** et lancez l'entraînement.")
    st.stop()

ml   = available["ml_results"]
arima = available["arima_result"] or {}
lstm  = available["lstm_result"] or {}
hyb   = available["hybrid_result"] or {}
rw    = available["rw_metrics"] or {}
horizon = st.session_state.get("horizon", 5)

# Dictionnaire unifié des modèles disponibles
models: dict[str, dict] = {}
if rw:
    models["Random Walk"] = {"metrics": rw, "color": "#64748b", "symbol": "●"}
if arima.get("metrics"):
    models["ARIMA"] = {**arima, "color": "#8b5cf6", "symbol": "◆"}
if ml.get("rf", {}).get("metrics"):
    models["RandomForest"] = {**ml["rf"], "color": "#f59e0b", "symbol": "▲"}
if ml.get("xgb") and ml["xgb"].get("metrics"):
    models["XGBoost"] = {**ml["xgb"], "color": "#ef4444", "symbol": "■"}
if lstm.get("metrics"):
    models["LSTM"] = {**lstm, "color": "#3b82f6", "symbol": "◉"}
if hyb.get("metrics"):
    models["Stacking"] = {**hyb, "color": "#22c55e", "symbol": "★"}

try:
    from models.lstm_model import diebold_mariano_test
    from models.hybrid_model import brier_score_multiclass, log_loss_multiclass
    from sklearn.preprocessing import LabelEncoder
    SCORING_OK = True
except ImportError:
    SCORING_OK = False


# ---------------------------------------------------------------------------
# ── 1. Tableau comparatif multi-métriques ─────────────────────────────────
# ---------------------------------------------------------------------------
st.subheader(f"📊 Comparaison complète — Horizon {horizon}j")

rows = []
le = LabelEncoder(); le.fit([-1, 0, 1])

for name, res in models.items():
    m = res.get("metrics", {})
    row = {
        "Modèle":        name,
        "DA (%)":        m.get("da_pct", np.nan),
        "Folds (moy DA)":np.mean(m.get("da_folds", [m.get("da_pct", np.nan)])),
        "RMSE":          m.get("rmse", np.nan),
        "MAE":           m.get("mae",  np.nan),
        "N échantillons":int(m.get("n_samples", 0)),
        "Brier ↓":       np.nan,
        "Log-Loss ↓":    np.nan,
    }
    # Brier + Log-Loss si probas disponibles
    if SCORING_OK and "probabilities" in res and "actuals" in res:
        try:
            probas   = np.array(res["probabilities"])
            actuals  = res["actuals"].values
            y_enc    = le.transform(actuals[:len(probas)])
            row["Brier ↓"]    = round(brier_score_multiclass(y_enc, probas[:len(y_enc)]), 4)
            row["Log-Loss ↓"] = round(log_loss_multiclass(y_enc, probas[:len(y_enc)]), 4)
        except Exception:
            pass
    rows.append(row)

df_bench = pd.DataFrame(rows)

def color_da(val):
    if pd.isna(val): return ""
    if val >= 57: return "background-color:#14532d; color:#4ade80; font-weight:800"
    if val >= 54: return "background-color:#166534; color:#86efac; font-weight:700"
    if val >= 51: return "color:#f59e0b"
    return "color:#ef4444"

def color_brier(val):
    if pd.isna(val): return ""
    if val <= 0.55: return "color:#22c55e; font-weight:700"
    if val <= 0.65: return "color:#f59e0b"
    return "color:#ef4444"

st.dataframe(
    df_bench.style
        .applymap(color_da, subset=["DA (%)", "Folds (moy DA)"])
        .applymap(color_brier, subset=["Brier ↓"])
        .format({
            "DA (%)": "{:.1f}",
            "Folds (moy DA)": "{:.1f}",
            "RMSE": "{:.5f}",
            "MAE":  "{:.5f}",
            "Brier ↓": "{:.4f}",
            "Log-Loss ↓": "{:.4f}",
        }, na_rep="—"),
    width="stretch",
    hide_index=True,
)

st.caption("""
**Lecture** : DA% > 55% = signal économiquement utile | Brier ↓ = meilleure calibration probabiliste |
Log-Loss ↓ = meilleure incertitude prédictive | toutes métriques calculées sur données **hors-échantillon**.
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── 2. Radar chart multi-critères ─────────────────────────────────────────
# ---------------------------------------------------------------------------
st.subheader("🕸️ Vue Radar — Profil de Chaque Modèle")

radar_axes = ["DA (%)", "Stabilité\nfolds", "Calibration\n(1-Brier)", "Rapidité\n(proxy)", "Interprét."]

interp_scores = {
    "Random Walk": 5, "ARIMA": 4, "RandomForest": 4, "XGBoost": 3, "LSTM": 2, "Stacking": 2
}
speed_scores = {
    "Random Walk": 5, "ARIMA": 4, "RandomForest": 3, "XGBoost": 3, "LSTM": 2, "Stacking": 1
}

fig_radar = go.Figure()

def _normalize(val, lo, hi):
    return max(0, min(5, (val - lo) / max(hi - lo, 1e-6) * 5))

for name, res in models.items():
    m = res.get("metrics", {})
    da      = m.get("da_pct", 50)
    folds   = m.get("da_folds", [da])
    stab    = 5 - _normalize(np.std(folds), 0, 15)   # stabilité inversée
    brier   = res.get("metrics", {}).get("brier", np.nan) if "brier" in m else np.nan
    calib   = (1 - float(brier)) * 5 if not np.isnan(brier) else 3.0
    speed   = speed_scores.get(name, 3)
    interp  = interp_scores.get(name, 3)
    da_norm = _normalize(da, 45, 70)

    vals = [da_norm, stab, calib, speed, interp]
    vals += [vals[0]]  # fermer le polygone

    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=radar_axes + [radar_axes[0]],
        fill="toself",
        name=name,
        line=dict(color=res["color"], width=2),
        opacity=0.65,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)",
    legend=dict(orientation="h", y=-0.1),
    margin=dict(l=20, r=20, t=20, b=40),
)
st.plotly_chart(fig_radar, width="stretch")
st.markdown("---")

# ---------------------------------------------------------------------------
# ── 3. Courbes d'apprentissage LSTM ────────────────────────────────────────
# ---------------------------------------------------------------------------
if lstm.get("learning_curves"):
    st.subheader("📉 Convergence LSTM — Train vs Validation Loss")
    curves = lstm["learning_curves"]

    fig_lc = make_subplots(
        rows=1, cols=len(curves),
        subplot_titles=[f"Fold {i+1}" for i in range(len(curves))],
        shared_yaxes=True,
    )
    for i, c in enumerate(curves):
        n = len(c.get("train", []))
        ep = list(range(n))
        fig_lc.add_trace(go.Scatter(
            x=ep, y=c["train"], mode="lines", name="Train" if i == 0 else None,
            line=dict(color="#3b82f6", width=2),
            showlegend=(i == 0),
        ), row=1, col=i + 1)
        fig_lc.add_trace(go.Scatter(
            x=list(range(len(c.get("val", [])))), y=c.get("val", []),
            mode="lines", name="Validation" if i == 0 else None,
            line=dict(color="#f59e0b", width=2, dash="dot"),
            showlegend=(i == 0),
        ), row=1, col=i + 1)

    fig_lc.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="#e2e8f0"),
    )
    fig_lc.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig_lc, width="stretch")
    st.caption(
        "Early stopping actif : l'entraînement s'arrête quand la val-loss stagne "
        "(patience configurable). Le gap train/val mesure le surapprentissage résiduel."
    )
    st.markdown("---")

# ---------------------------------------------------------------------------
# ── 4. Heatmap attention LSTM ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
if lstm.get("attention") and len(lstm["attention"]) > 0:
    st.subheader("🔍 Attention LSTM — Quels Timesteps sont Informatifs ?")

    try:
        # Moyenne des poids d'attention sur tous les folds + toutes les prédictions
        all_attn = np.vstack([a for fold in lstm["attention"] for a in fold])
        avg_attn = all_attn.mean(axis=0)     # (seq_len,)
        seq_len  = len(avg_attn)

        fig_attn = go.Figure(go.Bar(
            x=list(range(-seq_len + 1, 1)),
            y=avg_attn[::-1] if len(avg_attn) > 0 else avg_attn,
            marker=dict(
                color=avg_attn[::-1],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Poids"),
            ),
        ))
        fig_attn.update_layout(
            xaxis_title="Lag (jours avant la prédiction)",
            yaxis_title="Poids d'attention moyen",
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            yaxis=dict(gridcolor="#1e293b"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_attn, width="stretch")
        peak_lag = -seq_len + 1 + int(np.argmax(avg_attn[::-1]))
        st.caption(
            f"Le modèle accorde le plus d'attention au lag **{abs(peak_lag)}j** avant la prédiction. "
            "Les pics d'attention correspondent généralement aux périodes de forte volatilité récente."
        )
    except Exception as e:
        logger.debug("Attention heatmap : %s", e)

    st.markdown("---")

# ---------------------------------------------------------------------------
# ── 5. Matrices de confusion normalisées ──────────────────────────────────
# ---------------------------------------------------------------------------
models_with_preds = {k: v for k, v in models.items()
                     if "predictions" in v and "actuals" in v
                     and k not in ("Random Walk",)}

if models_with_preds:
    st.subheader("🎲 Matrices de Confusion Normalisées")
    le2 = LabelEncoder(); le2.fit([-1, 0, 1])
    labels_str = ["Baissier (-1)", "Neutre (0)", "Haussier (+1)"]

    ncols = min(3, len(models_with_preds))
    nrows = (len(models_with_preds) + ncols - 1) // ncols
    names_list = list(models_with_preds.keys())

    fig_cm = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=names_list,
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for idx, (name, res) in enumerate(models_with_preds.items()):
        row, col = divmod(idx, ncols)
        try:
            preds   = res["predictions"].values
            actuals = res["actuals"].values
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(actuals, preds, labels=[-1, 0, 1], normalize="true")

            fig_cm.add_trace(go.Heatmap(
                z=cm,
                x=["↓ Bas", "→ Neutre", "↑ Haut"],
                y=["Bas", "Neutre", "Haut"],
                colorscale="Blues",
                showscale=False,
                zmin=0, zmax=1,
                text=[[f"{v:.0%}" for v in row_] for row_ in cm],
                texttemplate="%{text}",
            ), row=row + 1, col=col + 1)
        except Exception:
            pass

    fig_cm.update_layout(
        height=300 * nrows,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(fig_cm, width="stretch")
    st.caption(
        "Matrices normalisées par ligne (recall). La diagonale = taux de bonne classification. "
        "Un bon classifieur a une diagonale dominante."
    )
    st.markdown("---")

# ---------------------------------------------------------------------------
# ── 6. DA% par fold — stabilité temporelle ────────────────────────────────
# ---------------------------------------------------------------------------
st.subheader("📈 Stabilité Temporelle — DA% par Fold")

fig_folds = go.Figure()
max_folds = 0

for name, res in models.items():
    folds = res.get("metrics", {}).get("da_folds", [])
    if not folds:
        continue
    max_folds = max(max_folds, len(folds))
    fig_folds.add_trace(go.Scatter(
        x=[f"Fold {i+1}" for i in range(len(folds))],
        y=folds,
        mode="lines+markers+text",
        name=name,
        line=dict(color=res["color"], width=2),
        marker=dict(size=8),
        text=[f"{v:.0f}%" for v in folds],
        textposition="top center",
    ))

fig_folds.add_hline(y=50, line_dash="dash", line_color="#64748b", annotation_text="Random (50%)")
fig_folds.add_hline(y=55, line_dash="dot",  line_color="#22c55e",  annotation_text="Cible (55%)")
fig_folds.update_layout(
    height=320,
    yaxis=dict(range=[30, 80], title="DA (%)", gridcolor="#1e293b"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.8)",
    legend=dict(orientation="h", y=1.05),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_folds, width="stretch")
st.caption(
    "Un modèle robuste maintient une DA% stable d'un fold à l'autre. "
    "Une chute brutale sur les derniers folds signale une rupture de régime de marché."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# ── 7. Courbes de calibration (fiabilité probabiliste) ───────────────────
# ---------------------------------------------------------------------------
models_with_probas = {k: v for k, v in models.items()
                      if "probabilities" in v and "actuals" in v}

if models_with_probas:
    st.subheader("🎯 Calibration Probabiliste — Fiabilité des Confiances")
    st.caption(
        "Un modèle calibré prédit p=70% exactement dans 70% des cas. "
        "La diagonale parfaite = calibration parfaite (modèle Brier-optimal)."
    )

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#64748b", dash="dash"),
        name="Calibration parfaite",
    ))

    le3 = LabelEncoder(); le3.fit([-1, 0, 1])
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for name, res in models_with_probas.items():
        try:
            probas  = np.array(res["probabilities"])
            actuals = res["actuals"].values
            y_enc   = le3.transform(actuals[:len(probas)])
            # Classe haussier (idx 2)
            p_haussier = probas[:len(y_enc), 2]
            y_haussier = (y_enc == 2).astype(int)
            bin_frac_pos, bin_mean_pred = [], []
            for i in range(n_bins):
                mask = (p_haussier >= bin_edges[i]) & (p_haussier < bin_edges[i + 1])
                if mask.sum() >= 3:
                    bin_frac_pos.append(y_haussier[mask].mean())
                    bin_mean_pred.append(p_haussier[mask].mean())
            if bin_mean_pred:
                fig_cal.add_trace(go.Scatter(
                    x=bin_mean_pred, y=bin_frac_pos,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=res["color"], width=2),
                    marker=dict(size=6),
                ))
        except Exception:
            pass

    fig_cal.update_layout(
        height=300,
        xaxis=dict(title="Probabilité haussier prédite", range=[0, 1], gridcolor="#1e293b"),
        yaxis=dict(title="Fréquence réelle haussier",    range=[0, 1], gridcolor="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_cal, width="stretch")
    st.markdown("---")

# ---------------------------------------------------------------------------
# ── 8. Test de Diebold-Mariano (significativité statistique) ─────────────
# ---------------------------------------------------------------------------
if SCORING_OK:
    models_dm = {k: v for k, v in models.items()
                 if "predictions" in v and "actuals" in v and k != "Random Walk"}

    if len(models_dm) >= 2:
        st.subheader("📐 Test Diebold-Mariano — Significativité Statistique")
        st.caption(
            "H₀ : les deux modèles sont équivalents. p < 0.05 → différence significative. "
            "Méthode : DM avec correction Harvey-Newbold-Leybourne (1997)."
        )

        dm_matrix = pd.DataFrame(index=list(models_dm.keys()), columns=list(models_dm.keys()), dtype=object)

        for n1, r1 in models_dm.items():
            for n2, r2 in models_dm.items():
                if n1 == n2:
                    dm_matrix.loc[n1, n2] = "—"
                    continue
                try:
                    common = r1["actuals"].index.intersection(r2["actuals"].index)
                    e1 = np.abs(r1["predictions"].loc[common].values - r1["actuals"].loc[common].values)
                    e2 = np.abs(r2["predictions"].loc[common].values - r2["actuals"].loc[common].values)
                    dm = diebold_mariano_test(e1, e2)
                    p = dm["p_value"]
                    flag = "✅" if p < 0.05 else "·"
                    dm_matrix.loc[n1, n2] = f"p={p:.3f} {flag}"
                except Exception:
                    dm_matrix.loc[n1, n2] = "n/d"

        st.dataframe(dm_matrix, width="stretch")
        st.caption("✅ = significatif au seuil 5% | · = non-significatif")
        st.markdown("---")

# ---------------------------------------------------------------------------
# ── 9. Poids du méta-apprenant hybride ────────────────────────────────────
# ---------------------------------------------------------------------------
if hyb.get("meta_coefs") is not None and len(hyb.get("meta_coefs", [])) > 0:
    st.subheader("🔗 Contribution des Modèles de Base (Méta-Apprenant)")
    st.caption(
        "Coefficients du méta-apprenant LogisticRegression appris sur les probabilités OOF. "
        "Un coefficient élevé = le modèle de base est informatif pour le méta-classifieur."
    )
    try:
        coefs = np.array(hyb["meta_coefs"])
        if coefs.ndim == 2:
            # coefs shape: (3_classes, n_features=n_models*3)
            n_models_meta = coefs.shape[1] // 3
            model_names_meta = ["RF", "XGBoost", "LSTM"][:n_models_meta]
            class_names = ["Baissier", "Neutre", "Haussier"]

            for ci, cname in enumerate(class_names):
                if ci < coefs.shape[0]:
                    c_vals = coefs[ci]
                    fig_w = go.Figure()
                    palette = ["#f59e0b", "#ef4444", "#3b82f6"]
                    for mi, mname in enumerate(model_names_meta):
                        start = mi * 3
                        fig_w.add_trace(go.Bar(
                            name=mname,
                            x=class_names,
                            y=c_vals[start:start + 3],
                            marker_color=palette[mi % len(palette)],
                        ))
                    if ci == 0:
                        fig_w.update_layout(
                            barmode="group", height=240,
                            title=dict(text="Coefficients méta-apprenant par classe de sortie"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.8)",
                            yaxis=dict(gridcolor="#1e293b"),
                            legend=dict(orientation="h", y=1.1),
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig_w, width="stretch")
                        break
    except Exception as e:
        logger.debug("Poids méta-apprenant : %s", e)

    st.markdown("---")

# ---------------------------------------------------------------------------
# Note méthodologique
# ---------------------------------------------------------------------------
with st.expander("📚 Note méthodologique"):
    st.markdown(f"""
    ### Protocole de validation — GoldSignal V2

    **Données** : XAU/EUR (log-rendements) — source yfinance (GC=F + EURUSD=X)

    **Cible** : classification ternaire {{baissier, neutre, haussier}} à horizon **{horizon} jours**,
    seuil = ±{st.session_state.get('seuil_direction', 0.3):.1f}% de log-rendement cumulé.

    **Protocole** : Walk-forward expanding window — aucune information future ne filtre vers le passé.
    Normalisation MinMaxScaler ajustée sur le train de chaque fold uniquement.

    **Métriques** :
    - **DA%** (Directional Accuracy) : proportion de directions correctes
    - **Brier Score** (Zadrozny & Elkan 2002) : proper scoring rule, mesure la qualité des probabilités
    - **Log-Loss** : proper scoring rule logarithmique (pénalise les confiances erronées)
    - **Test DM** (Diebold-Mariano 1995, corr. Harvey-Newbold-Leybourne 1997)

    **Modèles** :
    | Modèle | Classe | Paramétrage |
    |--------|--------|-------------|
    | Random Walk | Baseline | prédiction = signe du dernier retour |
    | ARIMA | Classique | sélection ordre par AIC sur chaque fold train |
    | RandomForest | Ensemble | class_weight=balanced, min_samples_leaf=10 |
    | XGBoost | Gradient Boosting | multi:softprob, early_stopping=50 |
    | LSTM | Deep Learning | bi-LSTM + dot-product attention + early stopping |
    | Stacking | Méta-apprentissage | LogReg sur probas OOF (Wolpert 1992) |

    **Référence** : Wolpert (1992) *Stacked Generalization*. Neural Networks 5(2).
    Diebold & Mariano (1995) *Comparing Predictive Accuracy*. JBES 13(3).
    """)

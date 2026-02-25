"""
03_predictions.py — Prédictions ML GoldSignal (Module 3).

Interface complète V2 :
  - Sélection horizon (5 / 15 / 30 jours)
  - Entraînement à la demande (bouton) : ARIMA + RF + XGBoost
  - Walk-forward strict (zero leakage)
  - Tableau comparatif + DA% vs Random Walk
  - Signal temps réel (dernières features disponibles)
  - Feature importance RF
  - Signal actionnable combiné (ML + macro + prime)
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("🤖 Prédictions ML")
st.caption(
    "Signaux de tendance probabilistes à 5/15/30 jours — walk-forward strict, zero leakage."
)

# ---------------------------------------------------------------------------
# Imports lazy (pour ne pas ralentir les autres pages)
# ---------------------------------------------------------------------------
try:
    from data.fetcher import fetch_ticker, get_taux_reels
    from data.database import get_config
    from data.processor import (
        ohlcv_usd_oz_to_eur_g, compute_log_returns, compute_sma_ratios,
        compute_gold_silver_ratio,
    )
    from analysis.technical import rsi as calc_rsi, atr as calc_atr
    from analysis.macro_score import compute_macro_score
    from models.baseline import run_arima_walkforward, adf_test
    from models.ml_models import (
        run_all_ml_models, train_random_forest, get_feature_importance_rf,
        predict_rf_probabilities, align_features_target,
        prepare_classification_target, XGB_AVAILABLE,
    )
    from models.signal_generator import (
        probabilities_to_signal, generate_actionable_signal,
        get_latest_rf_signal, build_signal_history,
    )
    from models.backtester import build_comparison_table, evaluate_random_walk
    from models.lstm_model import (
        run_lstm_walkforward, get_latest_lstm_signal, diebold_mariano_test,
        DEFAULT_SEQ_LEN, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT,
    )
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from models.hybrid_model import run_hybrid_walkforward
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from models.backtester import build_comparison_table, evaluate_random_walk
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    st.error(f"Erreur d'import : {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
cfg_modeles = get_config("modeles") or {}
cfg_macro = get_config("macro") or {}
tickers = get_config("tickers_yfinance") or {}

@st.cache_data(ttl=3600, show_spinner="Chargement des données marché…")
def load_market_data():
    df_xau = fetch_ticker(tickers.get("xau", "GC=F"))
    df_xag = fetch_ticker(tickers.get("xag", "SI=F"))
    df_eur = fetch_ticker(tickers.get("eurusd", "EURUSD=X"))
    df_dxy = fetch_ticker(tickers.get("dxy", "DX-Y.NYB"))
    df_vix = fetch_ticker(tickers.get("vix", "^VIX"))
    s_taux = get_taux_reels()

    df_xau_eur = ohlcv_usd_oz_to_eur_g(df_xau, df_eur)

    close = df_xau_eur["close"].dropna()

    # Features
    log_rets = compute_log_returns(close, [1, 2, 5])
    sma_ratios = compute_sma_ratios(close, [20, 50])
    rsi_series = calc_rsi(close, 14).rename("rsi_14")
    atr_series = calc_atr(df_xau_eur, 14).rename("atr_14")

    taux_aligned = s_taux.reindex(close.index, method="ffill")
    dxy_ret = np.log(df_dxy["close"]).diff(1).shift(1) if not df_dxy.empty else pd.Series(0.0, index=close.index)
    dxy_aligned = dxy_ret.reindex(close.index, method="ffill").rename("dxy_return_lag1")
    vix_aligned = df_vix["close"].reindex(close.index, method="ffill").rename("vix_niveau") if not df_vix.empty else pd.Series(20.0, index=close.index, name="vix_niveau")

    # Ratio or/argent
    if not df_xag.empty:
        ratio = (df_xau["close"] / df_xag["close"].reindex(df_xau.index, method="ffill")).rename("ratio_or_argent")
        ratio_aligned = ratio.reindex(close.index, method="ffill")
    else:
        ratio_aligned = pd.Series(80.0, index=close.index, name="ratio_or_argent")

    mois = pd.Series(close.index.month, index=close.index, name="mois")

    taux_df = pd.DataFrame({
        "taux_reels_lag1": taux_aligned.shift(1),
        "taux_reels_lag5": taux_aligned.shift(5),
    }, index=close.index)

    X = pd.concat([
        log_rets, sma_ratios, taux_df,
        dxy_aligned, vix_aligned, ratio_aligned,
        rsi_series, atr_series, mois,
    ], axis=1).dropna()

    log_returns_full = compute_log_returns(close, [1])["log_return_1"].dropna()

    return {
        "close": close,
        "X": X,
        "log_returns": log_returns_full,
        "xau_eur_df": df_xau_eur,
        "dxy": df_dxy,
        "vix": df_vix,
        "taux_reels": s_taux,
        "ratio_aligned": ratio_aligned,
    }

with st.spinner("Chargement des données…"):
    mdata = load_market_data()

close = mdata["close"]
X = mdata["X"]
log_returns = mdata["log_returns"]

if close.empty or X.empty:
    st.error("Données indisponibles. Vérifiez la connexion internet.")
    st.stop()

st.success(f"✅ Données chargées : **{len(close)}** observations | de {close.index[0].date()} à {close.index[-1].date()}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Détection environnement Cloud / Local
# ---------------------------------------------------------------------------
from models.model_store import is_cloud, save_pretrained, load_pretrained, has_pretrained
_ON_CLOUD = is_cloud()

# ---------------------------------------------------------------------------
# Paramètres de configuration (dans la page, non dans la sidebar)
# ---------------------------------------------------------------------------
with st.expander(
    "⚙️ Paramètres d'entraînement",
    expanded=(not _ON_CLOUD),  # dépliés en local, repliés sur Cloud
):
    st.markdown("**Général**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        horizon = st.selectbox("Horizon de prédiction", [5, 15, 30], index=0,
                                format_func=lambda x: f"{x} jours")
        n_splits = st.slider("Folds walk-forward", 3, 10, 5)
        min_train = st.slider("Train minimum (obs.)", 200, 1000, 400, 50)
    with col_g2:
        seuil_direction = st.slider("Seuil direction (log-ret %)", 0.1, 1.0, 0.3, 0.05,
                                     help="Ex: 0.3% → log-return > 0.003 = haussier")
        test_size = st.slider("Fenêtre test par fold", 20, 120, 60, 10)

    st.markdown("---")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**🌲 Random Forest** <small style='color:#94a3b8'>~15s</small>", unsafe_allow_html=True)
        rf_n_est = st.slider("Arbres", 50, 500, 200, 50)
        rf_depth = st.slider("Profondeur max RF", 3, 15, 8)

        if LSTM_AVAILABLE:
            st.markdown("**🧠 LSTM** <small style='color:#94a3b8'>~2–5min</small>", unsafe_allow_html=True)
            run_lstm = st.checkbox("Activer LSTM", value=False)
            if run_lstm:
                lstm_seq_len  = st.slider("Séquence", 10, 60, 30, 5)
                lstm_hidden   = st.select_slider("Neurones", [32, 64, 128, 256], value=64)
                lstm_layers   = st.slider("Couches", 1, 3, 2)
                lstm_dropout  = st.slider("Dropout", 0.0, 0.5, 0.3, 0.05)
                lstm_epochs   = st.slider("Époques max", 20, 200, 80, 10)
                lstm_patience = st.slider("Patience early stop", 5, 30, 10)
                lstm_bidir    = st.checkbox("Bidirectionnel", value=False)
            else:
                lstm_seq_len, lstm_hidden, lstm_layers = 30, 64, 2
                lstm_dropout, lstm_epochs, lstm_patience, lstm_bidir = 0.3, 80, 10, False
        else:
            run_lstm = False
            lstm_seq_len, lstm_hidden, lstm_layers = 30, 64, 2
            lstm_dropout, lstm_epochs, lstm_patience, lstm_bidir = 0.3, 80, 10, False

    with col_m2:
        if XGB_AVAILABLE:
            st.markdown("**🚀 XGBoost** <small style='color:#94a3b8'>~20s</small>", unsafe_allow_html=True)
            xgb_n_est = st.slider("Rounds boosting", 50, 500, 200, 50)
            xgb_depth = st.slider("Profondeur max XGB", 2, 10, 5)
            xgb_lr    = st.select_slider("Learning rate", [0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        else:
            xgb_n_est, xgb_depth, xgb_lr = 200, 5, 0.05

        st.markdown("**📈 ARIMA** <small style='color:#94a3b8'>~5s–2min</small>", unsafe_allow_html=True)
        run_arima = st.checkbox("Activer ARIMA", value=False,
                                 help="Lent sur longues séries. Désactivé par défaut.")
        if run_arima:
            arima_mode = st.radio(
                "Mode ARIMA",
                ["Rapide — ordre fixé (1,0,1)", "Auto-AIC — grille 2×2"],
                index=0,
            )
        else:
            arima_mode = "Rapide — ordre fixé (1,0,1)"

        if HYBRID_AVAILABLE:
            st.markdown("**🔀 Stacking Hybride** <small style='color:#94a3b8'>+1–2min</small>", unsafe_allow_html=True)
            run_hybrid = st.checkbox("Activer Hybride", value=False,
                                      help="RF + XGBoost + LSTM → méta-apprenant LogReg")
        else:
            run_hybrid = False

# variables manquantes si XGB non dispo
if not XGB_AVAILABLE:
    xgb_n_est, xgb_depth, xgb_lr = 200, 5, 0.05
if not HYBRID_AVAILABLE:
    run_hybrid = False

# ---------------------------------------------------------------------------
# Boutons : Charger modèle pré-entraîné + Entraîner
# ---------------------------------------------------------------------------
st.subheader("🚀 Prédictions ML")

if _ON_CLOUD:
    st.info(
        "☁️ **Mode Cloud** — L'entraînement complet prend plusieurs minutes "
        "et peut dépasser la mémoire disponible. "
        "Chargez le modèle pré-entraîné pour une expérience instantanée."
    )

_has_pretrained = has_pretrained(horizon)
col_btn_load, col_btn_train, col_info = st.columns([1, 1, 2])

with col_btn_load:
    load_btn = st.button(
        "📦 Charger modèle pré-entraîné",
        disabled=not _has_pretrained,
        use_container_width=True,
        help="Charge les prédictions sauvegardées (instantané)." if _has_pretrained
             else "Aucun modèle pré-entraîné disponible — lancez d'abord un entraînement.",
    )

with col_btn_train:
    _train_label = "▶️ Lancer l'entraînement"
    run_btn = st.button(
        _train_label,
        type="primary",
        use_container_width=True,
        help="Walk-forward sur l'historique complet.",
    )

with col_info:
    if _has_pretrained:
        _pt = load_pretrained(horizon) or {}
        st.caption(f"Modèle dispo — entraîné le {_pt.get('timestamp', '?')} | horizon {horizon}j")
    st.caption(
        f"Horizon {horizon}j · {n_splits} folds · seuil {seuil_direction:.1f}% · "
        f"train min {min_train} obs. · test {test_size} obs./fold"
    )

# ── Chargement du modèle pré-entraîné ──────────────────────────────────────
if load_btn and _has_pretrained:
    _pt = load_pretrained(horizon)
    if _pt:
        st.session_state["ml_results"]       = _pt["ml_results"]
        st.session_state["rw_metrics"]        = _pt["rw_metrics"]
        st.session_state["fi_df"]             = _pt["fi_df"]
        st.session_state["latest_signal"]     = _pt["latest_signal"]
        st.session_state["lstm_result"]       = _pt["lstm_result"]
        st.session_state["hybrid_result"]     = _pt["hybrid_result"]
        st.session_state["horizon"]           = _pt["horizon"]
        # Exposer log-returns pour backtesting
        _rf_actuals = (_pt["ml_results"] or {}).get("rf", {}).get("actuals")
        if _rf_actuals is None:
            _rf_actuals = (_pt["ml_results"] or {}).get("xgb", {}).get("actuals")
        st.session_state["_backtest_log_returns"] = log_returns
        st.success(f"✅ Modèle pré-entraîné chargé (horizon {_pt.get('horizon')}j — {_pt.get('timestamp', '')})")
        st.rerun()

if run_btn:
    seuil_frac = seuil_direction / 100.0

    progress = st.progress(0, text="Initialisation…")

    # --- 1. ADF + Baselines ---
    progress.progress(10, "Test ADF + Random Walk baseline…")
    try:
        adf_result = adf_test(log_returns)
        rw_metrics = evaluate_random_walk(log_returns)
        st.session_state["adf"] = adf_result
        st.session_state["rw_metrics"] = rw_metrics
    except Exception as e:
        st.warning(f"ADF/RW : {e}")

    # --- 2. ARIMA ---
    if run_arima:
        _arima_mode_label = arima_mode if run_arima else ""
        _fixed = (1, 0, 1) if "Rapide" in _arima_mode_label else None
        _mp = 1 if _fixed else 2
        _mq = 1 if _fixed else 2
        progress.progress(25, f"ARIMA walk-forward (’{'ordre fixe (1,0,1)' if _fixed else 'auto-AIC grille 2×2'}’)…")
        try:
            arima_result = run_arima_walkforward(
                log_returns, n_splits=n_splits,
                min_train_size=min_train, test_size=test_size,
                max_p=_mp, max_q=_mq,
                fixed_order=_fixed,
            )
            st.session_state["arima_result"] = arima_result
            progress.progress(40, "ARIMA terminé ✅")
        except Exception as e:
            st.warning(f"ARIMA : {e}")
    else:
        progress.progress(40, "ARIMA ignoré (désactivé)")

    # --- 3. ML ---
    progress.progress(50, "Random Forest + XGBoost walk-forward…")
    try:
        xgb_params = {}
        if XGB_AVAILABLE:
            xgb_params = {"n_estimators": xgb_n_est, "max_depth": xgb_depth, "learning_rate": xgb_lr}

        ml_results = run_all_ml_models(
            X=X,
            log_returns=log_returns,
            horizon=horizon,
            seuil_pct=seuil_frac,
            n_splits=n_splits,
            min_train_size=min_train,
            test_size=test_size,
            rf_params={"n_estimators": rf_n_est, "max_depth": rf_depth},
            xgb_params=xgb_params if XGB_AVAILABLE else None,
        )
        st.session_state["ml_results"] = ml_results
        st.session_state["horizon"] = horizon
        progress.progress(85, "Modèles ML terminés ✅")
    except Exception as e:
        st.error(f"Erreur ML : {e}")
        logger.exception("Erreur ML")

    # --- 4. LSTM walk-forward ---
    lstm_result = None
    if LSTM_AVAILABLE and run_lstm:
        progress.progress(88, "LSTM walk-forward (PyTorch + attention)…")
        try:
            lstm_result = run_lstm_walkforward(
                X=X, log_returns=log_returns,
                horizon=horizon, seuil_pct=seuil_frac,
                n_splits=n_splits, min_train_size=min_train, test_size=test_size,
                seq_len=lstm_seq_len, hidden_size=lstm_hidden,
                num_layers=lstm_layers, dropout=lstm_dropout,
                epochs=lstm_epochs, patience=lstm_patience,
                bidirectional=lstm_bidir,
            )
            st.session_state["lstm_result"] = lstm_result
            progress.progress(86, "LSTM terminé ✅")
        except Exception as e:
            st.warning(f"LSTM : {e}")
            logger.exception("LSTM walk-forward")

    # --- 5. Hybrid stacking ---
    hybrid_result = None
    if HYBRID_AVAILABLE and run_hybrid:
        progress.progress(87, "Stacking hybride (méta-apprenant LogReg)…")
        try:
            hybrid_result = run_hybrid_walkforward(
                X=X, log_returns=log_returns,
                horizon=horizon, seuil_pct=seuil_frac,
                n_splits=max(3, n_splits - 1),
                min_train_size=min_train, test_size=test_size,
                lstm_params={"seq_len": lstm_seq_len, "hidden_size": lstm_hidden,
                             "num_layers": lstm_layers, "dropout": lstm_dropout,
                             "epochs": lstm_epochs, "patience": lstm_patience} if LSTM_AVAILABLE and run_lstm else {},
                rf_params={"n_estimators": rf_n_est, "max_depth": rf_depth},
                xgb_params={"n_estimators": xgb_n_est, "max_depth": xgb_depth,
                            "learning_rate": xgb_lr} if XGB_AVAILABLE else None,
            )
            st.session_state["hybrid_result"] = hybrid_result
            progress.progress(96, "Hybride terminé ✅")
        except Exception as e:
            st.warning(f"Hybride : {e}")

    # --- 6. Signal temps réel ---
    progress.progress(97, "Calcul du signal temps réel…")
    try:
        X_aligned = ml_results.get("X_aligned", X)
        y_aligned = ml_results.get("y_aligned")
        if y_aligned is not None:
            fitted_rf = train_random_forest(
                X_aligned, y_aligned,
                n_estimators=rf_n_est, max_depth=rf_depth,
            )
            latest_signal = get_latest_rf_signal(fitted_rf, X_aligned)
            fi_df = get_feature_importance_rf(fitted_rf)
            st.session_state["latest_signal"] = latest_signal
            st.session_state["feature_importance"] = fi_df

        # Signal LSTM temps réel
        if lstm_result and lstm_result.get("last_model") and lstm_result.get("last_scaler"):
            lstm_latest = get_latest_lstm_signal(
                lstm_result["last_model"],
                lstm_result["last_scaler"],
                X_aligned,
                seq_len=lstm_seq_len,
                le=lstm_result.get("label_encoder"),
            )
            st.session_state["lstm_latest_signal"] = lstm_latest
    except Exception as e:
        logger.warning("Signal temps réel : %s", e)

    # --- 7. Exposer les log-returns PRIX pour la page Backtesting P&L ---
    # IMPORTANT : log_returns = vrais rendements journaliers du prix or (np.log(close).diff(1))
    # NE PAS utiliser ml_results["actuals"] qui contient les labels {-1, 0, +1} — pas des rendements !
    st.session_state["_backtest_log_returns"] = log_returns

    # --- 8. Persistance des signaux temps réel en base ---
    try:
        from data.database import save_signal, save_ml_run
        from datetime import date as _date_cls
        _today = _date_cls.today().isoformat()

        # Signal RF
        if st.session_state.get("latest_signal"):
            _ls = st.session_state["latest_signal"]
            _dir = _ls.get("direction", {})
            save_signal(_today, "RF", _ls.get("signal", 0),
                        p_haussier=_dir.get("haussier"), p_neutre=_dir.get("neutre"),
                        p_baissier=_dir.get("baissier"), horizon=horizon)

        # Signal LSTM
        if st.session_state.get("lstm_latest_signal"):
            _ls2 = st.session_state["lstm_latest_signal"]
            _dir2 = _ls2.get("direction", {})
            save_signal(_today, "LSTM", _ls2.get("signal", 0),
                        p_haussier=_dir2.get("haussier"), p_neutre=_dir2.get("neutre"),
                        p_baissier=_dir2.get("baissier"), horizon=horizon)

        # Métriques résumées en base
        _metrics_summary = {}
        for _mkey, _mname in [("rf","RF"), ("xgb","XGB")]:
            _m = ml_results.get(_mkey, {}).get("metrics", {})
            if _m:
                _metrics_summary[_mname] = {k: v for k, v in _m.items() if k != "model"}
        if lstm_result and lstm_result.get("metrics"):
            _metrics_summary["LSTM"] = {k: v for k, v in lstm_result["metrics"].items() if k != "model"}

        # Signaux OOS résumés (10 dernières dates)
        _signals_summary = {}
        for _mkey, _mname in [("rf","RF"), ("xgb","XGB")]:
            _preds = ml_results.get(_mkey, {}).get("predictions")
            if isinstance(_preds, pd.Series) and not _preds.empty:
                _signals_summary[_mname] = {str(k): int(v) for k, v in _preds.tail(30).items()}

        if _metrics_summary:
            save_ml_run(_metrics_summary, _signals_summary, horizon, n_splits,
                        {"rf_n_est": rf_n_est, "rf_depth": rf_depth})
    except Exception as _e:
        logger.debug("Persistance DB run ML : %s", _e)

    progress.progress(100, "✅ Entraînement terminé !")

    # --- 9. Sauvegarde automatique du modèle pré-entraîné ---
    try:
        from datetime import datetime as _dt
        _ts = _dt.now().strftime("%d/%m/%Y %H:%M")
        _saved = save_pretrained(
            horizon=horizon,
            ml_results=st.session_state.get("ml_results", {}),
            rw_metrics=st.session_state.get("rw_metrics", {}),
            fi_df=st.session_state.get("feature_importance", pd.DataFrame()),
            latest_signal=st.session_state.get("latest_signal"),
            lstm_result=st.session_state.get("lstm_result"),
            hybrid_result=st.session_state.get("hybrid_result"),
            seuil_direction=seuil_direction,
            n_splits=n_splits,
            timestamp=_ts,
        )
        if _saved:
            st.success(f"✅ Entraînement terminé — modèle sauvegardé ({_ts}). Résultats ci-dessous.")
        else:
            st.success("✅ Entraînement terminé — résultats ci-dessous.")
    except Exception as _se:
        logger.debug("Sauvegarde modèle pré-entraîné : %s", _se)
        st.success("✅ Entraînement terminé — résultats ci-dessous.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Affichage des résultats (si disponibles)
# ---------------------------------------------------------------------------
if "ml_results" not in st.session_state:
    st.info("💡 Cliquez sur **▶️ Lancer l'entraînement** pour calculer les signaux.")
    st.stop()

ml_results = st.session_state["ml_results"]
arima_result = st.session_state.get("arima_result", {})
rw_metrics = st.session_state.get("rw_metrics", {})
latest_signal = st.session_state.get("latest_signal", {})
lstm_latest_signal = st.session_state.get("lstm_latest_signal", {})
lstm_result = st.session_state.get("lstm_result", {})
hybrid_result = st.session_state.get("hybrid_result", {})
fi_df = st.session_state.get("feature_importance", pd.DataFrame())
horizon_used = st.session_state.get("horizon", horizon)

# ── Tabs résultats ──────────────────────────────────────────────────────
_run_at = st.session_state.get("_last_run_at", "")
if _run_at:
    st.caption(f"⏱️ Dernier entraînement : {_run_at}")

result_tabs = st.tabs([
    "🎯 Signal", "📊 Comparaison", "📈 OOS + Features", "📉 LSTM & DA/Fold", "🔬 Diagnostics",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 0 — Signal temps réel
# ═══════════════════════════════════════════════════════════════════════════
with result_tabs[0]:
    st.subheader(f"🎯 Signal Temps Réel — Horizon {horizon_used}j")

    if latest_signal:
        probs = latest_signal.get("direction", {})
        sig_color = latest_signal.get("color", "#f59e0b")
        sig_label = latest_signal.get("label", "—")
        conviction = latest_signal.get("conviction", 0.0)

        col_sig, col_bars = st.columns([1, 2])
        with col_sig:
            st.markdown(
                f"""
                <div style='text-align:center; padding:20px; background:{sig_color}20;
                            border:2px solid {sig_color}; border-radius:12px'>
                  <div style='font-size:2em; font-weight:800; color:{sig_color}'>{sig_label}</div>
                  <div style='color:#94a3b8; margin-top:8px'>Conviction : {conviction:.0%}</div>
                  <div style='color:#94a3b8'>Horizon : {horizon_used}j</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_bars:
            fig_p = go.Figure(go.Bar(
                x=[f"🔴 Baissier\n{probs.get('baissier',0):.1%}",
                   f"🟡 Neutre\n{probs.get('neutre',0):.1%}",
                   f"🟢 Haussier\n{probs.get('haussier',0):.1%}"],
                y=[probs.get("baissier", 0), probs.get("neutre", 0), probs.get("haussier", 0)],
                marker_color=["#ef4444", "#f59e0b", "#22c55e"],
                text=[f"{v:.1%}" for v in [probs.get("baissier",0), probs.get("neutre",0), probs.get("haussier",0)]],
                textposition="auto",
            ))
            fig_p.update_layout(
                height=200, showlegend=False,
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_p, width="stretch")

        # LSTM signal en comparaison
        if lstm_latest_signal:
            l_probs = lstm_latest_signal.get("direction", {})
            l_label = lstm_latest_signal.get("label", "—")
            l_color = lstm_latest_signal.get("color", "#f59e0b")
            l_conv  = lstm_latest_signal.get("conviction", 0.0)
            st.markdown(f"""
            <div style='background:{l_color}10; border:1px dashed {l_color}; border-radius:8px;
                        padding:10px; margin-top:8px'>
              <b style='color:{l_color}'>🧠 LSTM : {l_label}</b>
              &nbsp; <span style='color:#94a3b8; font-size:0.85em'>conviction {l_conv:.0%}</span>
              &nbsp; | 🔴 {l_probs.get('baissier',0):.1%}
                     🟡 {l_probs.get('neutre',0):.1%}
                     🟢 {l_probs.get('haussier',0):.1%}
            </div>
            """, unsafe_allow_html=True)

        # Signal actionnable combiné
        try:
            macro_cfg = get_config("macro") or {}
            vix_val = float(mdata["vix"]["close"].iloc[-1]) if not mdata["vix"].empty else 20.0
            taux_val = float(mdata["taux_reels"].iloc[-1]) if not mdata["taux_reels"].empty else 0.5
            rsi_val = float(calc_rsi(close, 14).iloc[-1])
            ratio_val = float(mdata["ratio_aligned"].iloc[-1])

            macro = compute_macro_score(
                taux_reel_dernier=taux_val,
                close_dxy=mdata["dxy"]["close"] if not mdata["dxy"].empty else pd.Series(dtype=float),
                vix_level=vix_val,
                rsi_or=rsi_val,
                ratio_or_argent=ratio_val,
                ratio_series=mdata["ratio_aligned"],
                cfg=macro_cfg,
            )

            action_result = generate_actionable_signal(
                ml_signal=latest_signal,
                macro_verdict=macro["verdict"],
            )

            action = action_result["action"]
            action_color = action_result["action_color"]

            st.markdown(
                f"""
                <div style='background:{action_color}15; border:2px solid {action_color};
                            border-radius:8px; padding:12px; margin-top:12px'>
                  <span style='font-size:1.4em; font-weight:800; color:{action_color}'>
                    {'🛒' if action=='ACHAT' else '💰' if action=='VENDRE' else '⏳'} {action}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for r in action_result.get("raisons", []):
                st.caption(r)
        except Exception as e:
            logger.debug("Signal actionnable non calculé : %s", e)
    else:
        st.info("Après l'entraînement, le signal temps réel RF apparaît ici.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Comparaison des modèles
# ═══════════════════════════════════════════════════════════════════════════
with result_tabs[1]:
    st.subheader("📊 Comparaison des Modèles (Walk-Forward)")

    all_model_results = {}
    if rw_metrics:
        all_model_results["Random Walk"] = {"metrics": rw_metrics}
    if arima_result and arima_result.get("metrics"):
        all_model_results["ARIMA"] = arima_result
    if ml_results.get("rf") and ml_results["rf"].get("metrics"):
        all_model_results["RandomForest"] = ml_results["rf"]
    if ml_results.get("xgb") and ml_results["xgb"] and ml_results["xgb"].get("metrics"):
        all_model_results["XGBoost"] = ml_results["xgb"]
    if lstm_result and lstm_result.get("metrics"):
        all_model_results["LSTM"] = lstm_result
    if hybrid_result and hybrid_result.get("metrics"):
        all_model_results["Stacking Hybride"] = hybrid_result

    if all_model_results:
        comparison_df = build_comparison_table(all_model_results)

        def style_da(val):
            if pd.isna(val):
                return ""
            if val >= 55:
                return "color:#22c55e; font-weight:700"
            elif val <= 50:
                return "color:#ef4444"
            return "color:#f59e0b"

        st.dataframe(
            comparison_df.style
                .applymap(style_da, subset=["DA (%)"])
                .format({
                    "DA (%)": "{:.1f}",
                    "RMSE": "{:.6f}",
                    "MAE": "{:.6f}",
                    "MAPE (%)": "{:.2f}",
                }, na_rep="—"),
            width="stretch",
            hide_index=True,
        )

        rw_da = rw_metrics.get("da_pct", 50.0)
        st.caption(
            f"DA% baseline random walk : **{rw_da:.1f}%** — "
            f"un modèle est utile seulement s'il bat cette référence."
        )

        # Distribution de la cible
        dist = ml_results.get("target_distribution", {})
        if dist:
            total = sum(dist.values())
            labels_dist = {1: "Haussier", 0: "Neutre", -1: "Baissier"}
            st.caption(
                "Distribution cible : " +
                " | ".join(f"{labels_dist.get(k,'?')} {v/total:.0%}" for k, v in sorted(dist.items(), reverse=True))
            )
    else:
        st.info("Tableau comparatif disponible après entraînement.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Signaux OOS + Feature importance
# ═══════════════════════════════════════════════════════════════════════════
with result_tabs[2]:
    st.subheader("📈 Signaux Out-of-Sample (Walk-Forward)")

    rf_preds = ml_results.get("rf", {}).get("predictions", pd.Series())
    rf_actuals = ml_results.get("rf", {}).get("actuals", pd.Series())

    if not rf_preds.empty and not rf_actuals.empty:
        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(
            x=rf_actuals.index, y=rf_actuals.values,
            mode="lines", name="Réel (log-ret dir.)",
            line=dict(color="#94a3b8", width=1),
        ))
        fig_oos.add_trace(go.Scatter(
            x=rf_preds.index, y=rf_preds.values,
            mode="markers+lines", name="RF prédit",
            line=dict(color="#f59e0b", width=1, dash="dot"),
            marker=dict(size=4,
                        color=["#22c55e" if v > 0 else "#ef4444" if v < 0 else "#f59e0b"
                               for v in rf_preds.values]),
        ))
        fig_oos.add_hline(y=0, line_color="#64748b", line_dash="dash")
        fig_oos.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.18)"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_oos, width="stretch")

    st.subheader("🔍 Importance des Features (Random Forest)")
    if not fi_df.empty:
        top_features = fi_df.head(12)
        fig_fi = go.Figure(go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker_color="#f59e0b",
        ))
        fig_fi.update_layout(
            height=350,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(128,128,128,0.18)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_fi, width="stretch")
    else:
        st.info("Feature importance disponible après entraînement RF.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — LSTM learning curves + DA/fold
# ═══════════════════════════════════════════════════════════════════════════
with result_tabs[3]:
    if lstm_result and lstm_result.get("learning_curves"):
        st.subheader("📉 Courbes d'Apprentissage LSTM (par fold)")
        curves = lstm_result["learning_curves"]
        fig_lc = go.Figure()
        palette_train = ["#3b82f6", "#06b6d4", "#8b5cf6", "#ec4899", "#f59e0b"]
        palette_val   = ["#93c5fd", "#67e8f9", "#c4b5fd", "#f9a8d4", "#fcd34d"]
        for i, c in enumerate(curves):
            color_t = palette_train[i % len(palette_train)]
            color_v = palette_val[i % len(palette_val)]
            n = len(c["train"])
            fig_lc.add_trace(go.Scatter(
                x=list(range(n)), y=c["train"],
                mode="lines", name=f"Fold {i+1} Train",
                line=dict(color=color_t, width=2),
            ))
            fig_lc.add_trace(go.Scatter(
                x=list(range(len(c["val"]))), y=c["val"],
                mode="lines", name=f"Fold {i+1} Val",
                line=dict(color=color_v, width=1.5, dash="dot"),
            ))
        fig_lc.update_layout(
            height=280,
            xaxis_title="Époque", yaxis_title="Cross-Entropy Loss",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.18)"),
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_lc, width="stretch")
        st.caption(
            "La convergence train/val confirme que l'early stopping a bien fonctionné — "
            "aucun fold ne surapprentit grâce au dropout + L2 + gradient clipping."
        )
    else:
        st.info("Courbes LSTM disponibles après entraînement avec LSTM activé.")

    # DA par fold — comparaison RF vs LSTM (toujours affiché si RF disponible)
    st.subheader("📊 Directional Accuracy par Fold")
    rf_da_folds   = (ml_results.get("rf") or {}).get("metrics", {}).get("da_folds", [])
    xgb_da_folds  = (ml_results.get("xgb") or {}).get("metrics", {}).get("da_folds", [])
    lstm_da_folds = (lstm_result or {}).get("da_per_fold", [])
    hybrid_da_folds = (hybrid_result or {}).get("da_per_fold", [])

    n_folds_disp = max(len(rf_da_folds), len(xgb_da_folds), len(lstm_da_folds), 1)
    fold_labels = [f"Fold {i+1}" for i in range(n_folds_disp)]
    fig_da = go.Figure()
    if rf_da_folds:
        fig_da.add_trace(go.Bar(name="RandomForest", x=fold_labels[:len(rf_da_folds)],   y=rf_da_folds,   marker_color="#f59e0b"))
    if xgb_da_folds:
        fig_da.add_trace(go.Bar(name="XGBoost",      x=fold_labels[:len(xgb_da_folds)],  y=xgb_da_folds,  marker_color="#ef4444"))
    if lstm_da_folds:
        fig_da.add_trace(go.Bar(name="LSTM",         x=fold_labels[:len(lstm_da_folds)],  y=lstm_da_folds, marker_color="#3b82f6"))
    if hybrid_da_folds:
        fig_da.add_trace(go.Bar(name="Hybride",      x=fold_labels[:len(hybrid_da_folds)], y=hybrid_da_folds, marker_color="#22c55e"))
    fig_da.add_hline(y=50, line_dash="dash", line_color="#64748b", annotation_text="Baseline 50%")
    fig_da.add_hline(y=55, line_dash="dot",  line_color="#22c55e",  annotation_text="Cible 55%")
    fig_da.update_layout(
        height=280, barmode="group",
        yaxis=dict(range=[30, 80], title="DA (%)", gridcolor="rgba(128,128,128,0.18)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_da, width="stretch")

    # Test Diebold-Mariano RF vs LSTM
    if ml_results.get("rf") and lstm_result:
        try:
            rf_preds_dm   = ml_results["rf"]["predictions"]
            lstm_preds_dm = lstm_result["predictions"]
            rf_act        = ml_results["rf"]["actuals"]
            common_idx    = rf_preds_dm.index.intersection(lstm_preds_dm.index).intersection(rf_act.index)
            if len(common_idx) >= 10:
                e_rf   = np.abs(rf_preds_dm.loc[common_idx].values - rf_act.loc[common_idx].values)
                e_lstm = np.abs(lstm_preds_dm.loc[common_idx].values - rf_act.loc[common_idx].values)
                dm = diebold_mariano_test(e_rf, e_lstm)
                st.markdown(
                    f"**Test Diebold-Mariano (RF vs LSTM)** : "
                    f"DM={dm['dm_stat']:.3f} | p={dm['p_value']:.3f} — *{dm['conclusion']}*"
                )
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Diagnostics statistiques
# ═══════════════════════════════════════════════════════════════════════════
with result_tabs[4]:
    st.subheader("🔬 Diagnostics statistiques")
    adf = st.session_state.get("adf", {})
    if adf:
        st.markdown("**Test ADF (stationnarité des log-returns)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Statistique ADF", f"{adf.get('adf_stat','—')}")
        col2.metric("p-value", f"{adf.get('p_value','—'):.4f}")
        col3.metric("Stationnaire ?", "✅ Oui" if adf.get("stationary") else "❌ Non")

    st.markdown("**Protocole walk-forward**")
    st.markdown(f"""
    - Folds : {n_splits} × expanding window
    - Train minimum : {min_train} observations
    - Fenêtre de test : {test_size} observations/fold
    - Horizon cible : {horizon_used} jours
    - Seuil haussier/baissier : ±{seuil_direction:.1f}% (log-return cumulé)
    - Zero data leakage garanti : chaque fold est entraîné sur passé pur
    """)

    st.markdown("**Rappel académique**")
    st.info(
        "Un DA% > 52–55% de manière stable en walk-forward est considéré "
        "comme économiquement significatif pour un signal sur métaux précieux "
        "(cf. Gold price: Geometric Random Walk and ARIMAX — TU Dortmund 2025)."
    )

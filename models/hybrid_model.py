"""
hybrid_model.py — Ensemble par stacking (méta-apprentissage) GoldSignal.

Méthodologie :
  Niveau 1 (base learners) :
    - RandomForest  → probabilités out-of-fold (N, 3)
    - XGBoost       → probabilités out-of-fold (N, 3)
    - LSTMClassifier→ probabilités out-of-fold (N, 3)

  Niveau 2 (méta-apprenant) :
    - LogisticRegression(C=1.0) entraîné sur la concaténation des probas OOF (N, 9)
    - Calibration isotonique optionnelle (Zadrozny & Elkan 2002)

  Évaluation :
    - Walk-forward outer-fold (aucune information future)
    - Métriques identiques aux modèles N1 pour comparaison directe

  Référence académique :
    Wolpert (1992) — Stacked Generalization. Neural Networks.
    Gneiting & Raftery (2007) — Strictly Proper Scoring Rules.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers probabilistes
# ---------------------------------------------------------------------------
def _safe_proba(model, X: np.ndarray, classes: list = [-1, 0, 1]) -> np.ndarray:
    """Retourne les probas dans l'ordre [baissier, neutre, haussier] — même si
    certaines classes sont absentes du train fold."""
    raw = model.predict_proba(X)
    cls = list(model.classes_)
    out = np.zeros((len(X), 3), dtype=np.float32)
    for i, c in enumerate(classes):
        if c in cls:
            out[:, i] = raw[:, cls.index(c)]
    return out


def brier_score_multiclass(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Brier Score multi-classe (plus bas = meilleur, 0 = parfait)."""
    n, k = probas.shape
    y_oh = np.zeros_like(probas)
    for i, yi in enumerate(y_true):
        y_oh[i, yi] = 1.0
    return float(np.mean(np.sum((probas - y_oh) ** 2, axis=1)))


def log_loss_multiclass(y_true: np.ndarray, probas: np.ndarray, eps: float = 1e-7) -> float:
    """Log-loss multi-classe (proper scoring rule)."""
    p_clip = np.clip(probas, eps, 1 - eps)
    n = len(y_true)
    return float(-np.mean([np.log(p_clip[i, y_true[i]]) for i in range(n)]))


# ---------------------------------------------------------------------------
# Construction des features OOF (out-of-fold) pour le méta-apprenant
# ---------------------------------------------------------------------------
def build_oof_features(
    rf_result:   dict,
    xgb_result:  Optional[dict],
    lstm_result: Optional[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Construit la matrice OOF pour le méta-apprenant à partir des probas des modèles N1.

    Chaque modèle contribue 3 colonnes (p_bais, p_neutre, p_haussier).
    Seuls les indices communs sont conservés.

    Returns:
        X_meta : (N, k*3) — features du méta-apprenant
        y_meta : (N,)     — labels encodés communs
    """
    sources = []
    indices = None

    for name, result in [("RF", rf_result), ("XGB", xgb_result), ("LSTM", lstm_result)]:
        if result is None or "probabilities" not in result:
            continue
        probas = result["probabilities"]    # (N, 3)
        preds  = result["predictions"]      # pd.Series avec index
        if isinstance(preds, pd.Series):
            idx = preds.index
            sources.append((name, probas, idx))
            indices = idx if indices is None else indices.intersection(idx)

    if not sources or indices is None or len(indices) == 0:
        raise ValueError("Pas assez de données OOF communes entre les modèles.")

    arrays = []
    for name, probas, idx in sources:
        # Ré-indexer sur l'intersection
        mask = [list(idx).index(i) for i in indices if i in list(idx)]
        if len(mask) != len(indices):
            logger.warning("[Hybrid] %s — indices OOF incomplets, alignement approximatif", name)
        arrays.append(probas[:len(indices)])

    X_meta = np.hstack(arrays)   # (N, n_models * 3)

    # Labels : utiliser les actuals du premier modèle disponible
    y_series = sources[0][1]     # probas du premier (on veut actuals)
    actuals = rf_result.get("actuals", None)
    if actuals is None:
        raise ValueError("Actuals manquants dans rf_result")

    le = LabelEncoder()
    le.fit([-1, 0, 1])
    y_meta = le.transform(actuals.values[:len(indices)])

    return X_meta, y_meta


# ---------------------------------------------------------------------------
# Méta-apprenant : LogisticRegression calibrée
# ---------------------------------------------------------------------------
def train_meta_learner(
    X_meta:    np.ndarray,
    y_meta:    np.ndarray,
    calibrate: bool = True,
    C:         float = 1.0,
) -> LogisticRegression:
    """Entraîne un méta-apprenant LogisticRegression sur les probas OOF.

    Optionnellement calibré par Platt scaling (cv="prefit" + CalibratedClassifierCV
    n'est pas disponible sans val — on utilise le calibrage intégré LR softmax).

    Args:
        X_meta:    (N, k*3) probas concaténées.
        y_meta:    (N,) labels encodés.
        calibrate: Si True, applique une régression logistique avec `multi_class='multinomial'`
                   qui est déjà une forme de calibration probabiliste.
        C:         Inverse de la régularisation L2.

    Returns:
        Méta-apprenant entraîné.
    """
    meta = LogisticRegression(
        C           = C,
        max_iter    = 1000,
        multi_class = "multinomial",
        solver      = "lbfgs",
        class_weight = "balanced",
    )
    meta.fit(X_meta, y_meta)
    logger.info("[Hybrid] Méta-apprenant entraîné | coeff shape=%s", meta.coef_.shape)
    return meta


# ---------------------------------------------------------------------------
# Walk-forward hybride complet
# ---------------------------------------------------------------------------
def run_hybrid_walkforward(
    X:              pd.DataFrame,
    log_returns:    pd.Series,
    horizon:        int   = 5,
    seuil_pct:      float = 0.003,
    n_splits:       int   = 5,
    min_train_size: int   = 400,
    test_size:      int   = 60,
    lstm_params:    Optional[dict] = None,
    rf_params:      Optional[dict] = None,
    xgb_params:     Optional[dict] = None,
) -> dict:
    """Entraîne les 3 modèles N1 + méta-apprenant en walk-forward strict.

    Protocole :
      Pour chaque fold outer [train_outer | test_outer] :
        1. Diviser train_outer en inner folds pour obtenir probas OOF
        2. Entraîner méta-apprenant sur OOF
        3. Ré-entraîner N1 sur tout train_outer
        4. Évaluer méta sur test_outer

    Note : protocole simplifié "single-level OOF" pour éviter le triple nesting.
    La version complète nécessiterait un CV nested (O(k²) entraînements).

    Returns:
        Dict avec metrics, predictions, actuals, model_weights (coeff LR).
    """
    from models.ml_models import (
        run_all_ml_models, prepare_classification_target,
        align_features_target, train_random_forest,
        predict_rf_probabilities, XGB_AVAILABLE,
    )
    from models.lstm_model import run_lstm_walkforward, DEFAULT_SEQ_LEN
    from models.backtester import WalkForwardCV, compute_metrics

    lstm_p = lstm_params  or {}
    rf_p   = rf_params    or {}
    xgb_p  = xgb_params   or {}

    # --- Cible ---
    from models.ml_models import prepare_classification_target, align_features_target
    y = prepare_classification_target(log_returns, horizon=horizon, seuil_pct=seuil_pct)
    X_al, y_al = align_features_target(X, y)

    le = LabelEncoder()
    le.fit([-1, 0, 1])

    cv = WalkForwardCV(n_splits=n_splits, min_train_size=min_train_size, test_size=test_size)
    splits = cv.split(X_al)

    if not splits:
        raise ValueError("[Hybrid] Pas assez de données.")

    all_preds_meta   = []
    all_actuals_meta = []
    all_probas_meta  = []
    da_per_fold      = []
    meta_coefs       = []
    last_meta        = None

    logger.info("[Hybrid] Début walk-forward | %d folds", len(splits))

    X_np = X_al.values.astype(np.float32)
    y_np = y_al.values

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info("[Hybrid] Fold %d/%d", fold_idx + 1, len(splits))

        X_tr = X_al.iloc[train_idx]
        X_te = X_al.iloc[test_idx]
        y_tr = y_al.iloc[train_idx]
        y_te = y_al.iloc[test_idx]

        # ── Niveau 1 : RF walk-forward sur train_fold pour OOF ──
        inner_n_splits = max(3, n_splits - 1)
        try:
            ml_oof = run_all_ml_models(
                X=X_tr, log_returns=log_returns.loc[y_tr.index],
                horizon=horizon, seuil_pct=seuil_pct,
                n_splits=inner_n_splits,
                min_train_size=max(100, min_train_size // 2),
                test_size=max(20, test_size // 2),
                rf_params=rf_p, xgb_params=xgb_p if XGB_AVAILABLE else None,
            )
        except Exception as exc:
            logger.warning("[Hybrid] Fold %d — OOF ML erreur : %s", fold_idx + 1, exc)
            continue

        # ── LSTM OOF sur train_fold ──
        try:
            lstm_oof = run_lstm_walkforward(
                X=X_tr, log_returns=log_returns.loc[y_tr.index],
                horizon=horizon, seuil_pct=seuil_pct,
                n_splits=inner_n_splits,
                min_train_size=max(100, min_train_size // 2),
                test_size=max(20, test_size // 2),
                **lstm_p,
            )
        except Exception as exc:
            logger.warning("[Hybrid] Fold %d — OOF LSTM erreur : %s", fold_idx + 1, exc)
            lstm_oof = None

        # ── Construction features OOF & méta-apprenant ──
        try:
            X_meta_oof, y_meta_oof = _build_meta_features_simple(
                ml_oof["rf"], ml_oof.get("xgb"), lstm_oof, y_tr
            )
            meta = train_meta_learner(X_meta_oof, y_meta_oof)
            meta_coefs.append(meta.coef_)
        except Exception as exc:
            logger.warning("[Hybrid] Fold %d — méta-apprenant erreur : %s", fold_idx + 1, exc)
            continue

        # ── Ré-entraîner N1 sur tout train_fold (pas OOF) ──
        from sklearn.ensemble import RandomForestClassifier
        rf_full = RandomForestClassifier(
            n_estimators=rf_p.get("n_estimators", 200),
            max_depth=rf_p.get("max_depth", 8),
            class_weight="balanced",
            n_jobs=-1,
        )
        y_tr_enc = le.transform(y_tr.values)
        rf_full.fit(X_tr.values, y_tr_enc)

        xgb_full = None
        if XGB_AVAILABLE:
            try:
                from xgboost import XGBClassifier
                xgb_full = XGBClassifier(
                    n_estimators=xgb_p.get("n_estimators", 200),
                    max_depth=xgb_p.get("max_depth", 5),
                    learning_rate=xgb_p.get("learning_rate", 0.05),
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                )
                xgb_full.fit(X_tr.values, y_tr_enc)
            except Exception as exc:
                logger.warning("[Hybrid] XGBoost full fit erreur : %s", exc)

        # LSTM sur tout train
        lstm_full = None
        lstm_scaler_full = None
        try:
            from models.lstm_model import _train_fold as lstm_train, DEFAULT_SEQ_LEN, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT
            seq_len = lstm_p.get("seq_len", DEFAULT_SEQ_LEN)
            val_split = max(seq_len + 1, int(len(X_tr) * 0.1))
            counts_tr = np.bincount(y_tr_enc, minlength=3)
            counts_tr = np.maximum(counts_tr, 1)
            import torch
            cw = torch.tensor(len(y_tr_enc) / (3.0 * counts_tr), dtype=torch.float32)
            lstm_full, _, _, lstm_scaler_full = lstm_train(
                X_tr.values[:-val_split], y_tr_enc[:-val_split],
                X_tr.values[-val_split:], y_tr_enc[-val_split:],
                seq_len=seq_len,
                hidden_size=lstm_p.get("hidden_size", DEFAULT_HIDDEN_SIZE),
                num_layers=lstm_p.get("num_layers", DEFAULT_NUM_LAYERS),
                dropout=lstm_p.get("dropout", DEFAULT_DROPOUT),
                class_weights=cw,
            )
        except Exception as exc:
            logger.warning("[Hybrid] Fold %d — LSTM full fit erreur : %s", fold_idx + 1, exc)

        # ── Prédictions N1 sur test ──
        X_te_np = X_te.values.astype(np.float32)

        rf_te_probas = _safe_proba_enc(rf_full, X_te_np, le)
        xgb_te_probas = _safe_proba_enc(xgb_full, X_te_np, le) if xgb_full else np.full((len(X_te), 3), 1/3)
        lstm_te_probas = _get_lstm_probas(lstm_full, lstm_scaler_full, X_te_np,
                                           lstm_p.get("seq_len", 30))

        # Aligner la taille (LSTM perd seq_len observations)
        min_len = min(len(rf_te_probas), len(xgb_te_probas), len(lstm_te_probas))
        offset  = max(len(rf_te_probas) - min_len, 0)

        rf_te_al   = rf_te_probas[-min_len:]
        xgb_te_al  = xgb_te_probas[-min_len:]
        lstm_te_al = lstm_te_probas[-min_len:]
        y_te_al    = le.transform(y_te.values[-min_len:])
        idx_te_al  = y_te.index[-min_len:]

        X_meta_test = np.hstack([rf_te_al, xgb_te_al, lstm_te_al])
        meta_probas = meta.predict_proba(X_meta_test)     # (N, 3)
        meta_preds  = np.argmax(meta_probas, axis=1)

        preds_orig   = le.inverse_transform(meta_preds)
        actuals_orig = le.inverse_transform(y_te_al)

        da = np.mean(preds_orig == actuals_orig) * 100
        da_per_fold.append(da)
        logger.info("[Hybrid] Fold %d — DA=%.1f%%", fold_idx + 1, da)

        all_preds_meta.append(pd.Series(preds_orig, index=idx_te_al))
        all_actuals_meta.append(pd.Series(actuals_orig, index=idx_te_al))
        all_probas_meta.append(meta_probas)
        last_meta = meta

    if not all_preds_meta:
        raise RuntimeError("[Hybrid] Aucun fold valide.")

    preds_series   = pd.concat(all_preds_meta)
    actuals_series = pd.concat(all_actuals_meta)
    probas_arr     = np.vstack(all_probas_meta)

    da_global = np.mean(preds_series.values == actuals_series.values) * 100

    # Poids moyens du méta-apprenant (interprétabilité)
    avg_coefs = np.mean(meta_coefs, axis=0) if meta_coefs else np.array([[]])

    metrics = {
        "da_pct":    round(da_global, 2),
        "da_folds":  [round(d, 2) for d in da_per_fold],
        "n_samples": len(preds_series),
        "n_folds":   len(da_per_fold),
        "rmse": float(np.sqrt(np.mean((preds_series.values - actuals_series.values) ** 2))),
        "mae":  float(np.mean(np.abs(preds_series.values - actuals_series.values))),
        "brier": brier_score_multiclass(
            le.transform(actuals_series.values), probas_arr
        ),
    }

    logger.info("[Hybrid] Global — DA=%.1f%% | Brier=%.4f", da_global, metrics["brier"])

    return {
        "metrics":         metrics,
        "predictions":     preds_series,
        "actuals":         actuals_series,
        "probabilities":   probas_arr,
        "da_per_fold":     da_per_fold,
        "meta_coefs":      avg_coefs,
        "last_meta":       last_meta,
        "label_encoder":   le,
    }


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def _build_meta_features_simple(
    rf_result:   dict,
    xgb_result:  Optional[dict],
    lstm_result: Optional[dict],
    y_ref:       pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit features OOF homogènes pour le méta-apprenant (version simplifiée).

    Aligne les probas OOF sur la taille minimale commune.
    """
    le = LabelEncoder()
    le.fit([-1, 0, 1])

    parts = []
    min_n = None

    for result in [rf_result, xgb_result, lstm_result]:
        if result is None or "probabilities" not in result:
            continue
        probas = np.array(result["probabilities"])
        if probas.ndim == 2 and probas.shape[1] == 3:
            parts.append(probas)
            min_n = len(probas) if min_n is None else min(min_n, len(probas))

    if not parts:
        raise ValueError("Aucune proba OOF disponible.")

    parts = [p[-min_n:] for p in parts]
    X_meta = np.hstack(parts)
    y_meta = le.transform(y_ref.values[-min_n:])
    return X_meta, y_meta


def _safe_proba_enc(model, X: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """Retourne (N, 3) dans l'ordre [baissier, neutre, haussier] avec LabelEncoder."""
    if model is None:
        return np.full((len(X), 3), 1 / 3, dtype=np.float32)
    raw = model.predict_proba(X)
    classes_enc = list(model.classes_)               # [0, 1, 2] ou sous-ensemble
    classes_orig = le.inverse_transform(classes_enc)  # [-1, 0, 1]
    out = np.full((len(X), 3), 1e-6, dtype=np.float32)
    for enc_idx, orig in zip(range(len(classes_orig)), classes_orig):
        target = list(le.classes_).index(orig)
        out[:, target] = raw[:, enc_idx]
    # Normaliser
    out /= out.sum(axis=1, keepdims=True)
    return out


def _get_lstm_probas(
    model, scaler, X_test: np.ndarray, seq_len: int = 30
) -> np.ndarray:
    """Prédictions LSTM glissantes sur X_test normalisé."""
    if model is None or scaler is None:
        return np.full((len(X_test), 3), 1 / 3, dtype=np.float32)

    from models.lstm_model import _predict_sequence
    X_s = scaler.transform(X_test)
    _, probas, _ = _predict_sequence(model, X_s, seq_len)

    if len(probas) == 0:
        return np.full((len(X_test), 3), 1 / 3, dtype=np.float32)
    return probas.astype(np.float32)

"""
ml_models.py — Modèles ML supervisés pour GoldSignal.

Implémente :
  - Random Forest (scikit-learn)
  - XGBoost

Cible : classification ternaire de la direction (haussier / neutre / baissier)
        OU régression sur le log-return (selon le mode choisi).

Protocole : walk-forward validation strict (backtester.py).
Features : matrice construite par data/processor.py.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.getLogger(__name__).warning("XGBoost non installé. Modèle XGBoost désactivé.")

from models.backtester import WalkForwardCV, compute_metrics, build_comparison_table

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers : préparation des données
# ---------------------------------------------------------------------------

def prepare_classification_target(log_returns: pd.Series,
                                    horizon: int = 5,
                                    seuil_pct: float = 0.3) -> pd.Series:
    """Crée la cible ternaire de direction.

    - +1 : somme des log-returns sur l'horizon > seuil
    - -1 : somme < -seuil
    -  0 : neutre

    Args:
        log_returns: Série de log-rendements journaliers.
        horizon: Fenêtre de prédiction en jours.
        seuil_pct: Seuil en fraction de log-return (ex: 0.003 = ~0.3%).

    Returns:
        pd.Series de {-1, 0, 1}.
    """
    future_ret = log_returns.rolling(horizon).sum().shift(-horizon)
    direction = future_ret.apply(
        lambda r: 1 if r > seuil_pct else (-1 if r < -seuil_pct else 0)
        if not np.isnan(r) else np.nan
    )
    direction.name = f"direction_{horizon}j"
    return direction


def align_features_target(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Aligne features et cible sur l'index commun sans NaN.

    Args:
        X: Matrice de features.
        y: Série cible.

    Returns:
        Tuple (X_clean, y_clean) alignés et sans NaN.
    """
    combined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    X_clean = combined.drop(columns=["__target__"])
    y_clean = combined["__target__"]
    return X_clean, y_clean


# ---------------------------------------------------------------------------
# Random Forest — Classification
# ---------------------------------------------------------------------------

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                         n_estimators: int = 200, max_depth: int = 8,
                         random_state: int = 42) -> dict:
    """Entraîne un Random Forest Classifier.

    Args:
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement ({-1, 0, 1}).
        n_estimators: Nombre d'arbres.
        max_depth: Profondeur maximale.
        random_state: Graine pour reproductibilité.

    Returns:
        Dict avec 'model' (classifier) et 'feature_names'.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train.values, y_train.values.astype(int))
    return {"model": clf, "feature_names": list(X_train.columns)}


def predict_random_forest(fitted: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Prédit la classe + retourne la prédiction en valeur numérique.

    Args:
        fitted: Résultat de train_random_forest().
        X_test: Features de test.

    Returns:
        np.ndarray des classes prédites {-1, 0, 1}.
    """
    return fitted["model"].predict(X_test.values).astype(float)


def predict_rf_probabilities(fitted: dict, X_test: pd.DataFrame) -> pd.DataFrame:
    """Retourne les probabilités des 3 classes.

    Args:
        fitted: Résultat de train_random_forest().
        X_test: Features de test.

    Returns:
        DataFrame avec colonnes ['p_baissier', 'p_neutre', 'p_haussier'].
        Les colonnes correspondent aux classes triées du classifier.
    """
    clf = fitted["model"]
    proba = clf.predict_proba(X_test.values)
    classes = clf.classes_
    df_proba = pd.DataFrame(proba, index=X_test.index, columns=[f"class_{int(c)}" for c in classes])

    # Renommage standardisé
    rename = {f"class_{-1}": "p_baissier", f"class_0": "p_neutre", f"class_1": "p_haussier"}
    return df_proba.rename(columns={k: v for k, v in rename.items() if k in df_proba.columns})


def get_feature_importance_rf(fitted: dict) -> pd.DataFrame:
    """Retourne l'importance des features du Random Forest.

    Args:
        fitted: Résultat de train_random_forest().

    Returns:
        DataFrame trié par importance décroissante.
    """
    importances = fitted["model"].feature_importances_
    return pd.DataFrame({
        "feature": fitted["feature_names"],
        "importance": importances,
    }).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# XGBoost — Classification
# ---------------------------------------------------------------------------

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                   n_estimators: int = 200, max_depth: int = 5,
                   learning_rate: float = 0.05,
                   random_state: int = 42) -> dict:
    """Entraîne un XGBoost Classifier pour classification ternaire.

    Args:
        X_train: Features d'entraînement.
        y_train: Cible {-1, 0, 1}.
        n_estimators: Nombre de boosting rounds.
        max_depth: Profondeur max des arbres.
        learning_rate: Taux d'apprentissage.
        random_state: Graine.

    Returns:
        Dict avec 'model', 'label_encoder', 'feature_names'.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost n'est pas installé.")

    # XGBoost requiert des labels dans [0, n_classes-1]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train.values.astype(int))

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",
        random_state=random_state,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X_train.values, y_encoded)
    return {"model": clf, "label_encoder": le, "feature_names": list(X_train.columns)}


def predict_xgboost(fitted: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Prédit les classes (converties en {-1, 0, 1}).

    Args:
        fitted: Résultat de train_xgboost().
        X_test: Features de test.

    Returns:
        np.ndarray des classes originales {-1, 0, 1}.
    """
    preds_encoded = fitted["model"].predict(X_test.values)
    return fitted["label_encoder"].inverse_transform(preds_encoded).astype(float)


def predict_xgb_probabilities(fitted: dict, X_test: pd.DataFrame) -> pd.DataFrame:
    """Retourne les probabilités des 3 classes XGBoost.

    Args:
        fitted: Résultat de train_xgboost().
        X_test: Features de test.

    Returns:
        DataFrame avec colonnes ['p_baissier', 'p_neutre', 'p_haussier'].
    """
    proba = fitted["model"].predict_proba(X_test.values)
    le = fitted["label_encoder"]
    classes = le.inverse_transform(np.arange(len(le.classes_)))

    df_proba = pd.DataFrame(proba, index=X_test.index, columns=[int(c) for c in classes])
    rename = {-1: "p_baissier", 0: "p_neutre", 1: "p_haussier"}
    return df_proba.rename(columns={k: v for k, v in rename.items() if k in df_proba.columns})


# ---------------------------------------------------------------------------
# Walk-Forward wrappers
# ---------------------------------------------------------------------------

def run_rf_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 400,
    test_size: int = 60,
    n_estimators: int = 200,
    max_depth: int = 8,
) -> dict:
    """Walk-forward validation pour Random Forest.

    Args:
        X: Matrice de features (DatetimeIndex).
        y: Cible ternaire {-1, 0, 1}.
        n_splits: Folds.
        min_train_size: Train minimum.
        test_size: Fenêtre de test.
        n_estimators: Arbres RF.
        max_depth: Profondeur max.

    Returns:
        Dict résultat walk-forward (voir WalkForwardCV.run).
    """
    def train_fn(X_tr, y_tr):
        return train_random_forest(X_tr, y_tr, n_estimators, max_depth)

    cv = WalkForwardCV(n_splits=n_splits, min_train_size=min_train_size, test_size=test_size)
    return cv.run(X, y, train_fn=train_fn, predict_fn=predict_random_forest, model_name="RandomForest")


def run_xgb_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 400,
    test_size: int = 60,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> dict:
    """Walk-forward validation pour XGBoost.

    Args:
        X: Matrice de features.
        y: Cible ternaire {-1, 0, 1}.
        n_splits: Folds.
        min_train_size: Train minimum.
        test_size: Fenêtre de test.
        n_estimators: Rounds de boosting.
        max_depth: Profondeur max.
        learning_rate: Learning rate.

    Returns:
        Dict résultat walk-forward.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost non disponible.")

    def train_fn(X_tr, y_tr):
        return train_xgboost(X_tr, y_tr, n_estimators, max_depth, learning_rate)

    cv = WalkForwardCV(n_splits=n_splits, min_train_size=min_train_size, test_size=test_size)
    return cv.run(X, y, train_fn=train_fn, predict_fn=predict_xgboost, model_name="XGBoost")


# ---------------------------------------------------------------------------
# Pipeline complet : entraîner tous les modèles ML sur les données disponibles
# ---------------------------------------------------------------------------

def run_all_ml_models(
    X: pd.DataFrame,
    log_returns: pd.Series,
    horizon: int = 5,
    seuil_pct: float = 0.003,
    n_splits: int = 5,
    min_train_size: int = 400,
    test_size: int = 60,
    rf_params: Optional[dict] = None,
    xgb_params: Optional[dict] = None,
) -> dict:
    """Entraîne RF + XGBoost et retourne les résultats walk-forward.

    Args:
        X: Matrice de features complète.
        log_returns: Série de log-rendements.
        horizon: Horizon de prédiction (jours).
        seuil_pct: Seuil pour définir haussier/baissier.
        n_splits: Folds walk-forward.
        min_train_size: Train minimum.
        test_size: Fenêtre de test.
        rf_params: Hyperparamètres RF (optionnel).
        xgb_params: Hyperparamètres XGBoost (optionnel).

    Returns:
        Dict avec :
          - 'rf': résultat walk-forward RF
          - 'xgb': résultat walk-forward XGBoost (None si non disponible)
          - 'target_distribution': distribution de la cible
          - 'comparison': DataFrame comparatif
          - 'horizon': horizon utilisé
    """
    rf_p = rf_params or {}
    xgb_p = xgb_params or {}

    # Cible ternaire
    y = prepare_classification_target(log_returns, horizon=horizon, seuil_pct=seuil_pct)
    X_aligned, y_aligned = align_features_target(X, y)

    logger.info(
        "[ML] Dataset : %d obs | horizon : %dj | classes : %s",
        len(X_aligned), horizon,
        y_aligned.value_counts().to_dict(),
    )

    # Random Forest
    logger.info("[ML] Random Forest walk-forward…")
    rf_result = run_rf_walkforward(
        X_aligned, y_aligned,
        n_splits=n_splits, min_train_size=min_train_size, test_size=test_size,
        n_estimators=rf_p.get("n_estimators", 200),
        max_depth=rf_p.get("max_depth", 8),
    )

    # XGBoost
    xgb_result = None
    if XGB_AVAILABLE:
        logger.info("[ML] XGBoost walk-forward…")
        try:
            xgb_result = run_xgb_walkforward(
                X_aligned, y_aligned,
                n_splits=n_splits, min_train_size=min_train_size, test_size=test_size,
                n_estimators=xgb_p.get("n_estimators", 200),
                max_depth=xgb_p.get("max_depth", 5),
                learning_rate=xgb_p.get("learning_rate", 0.05),
            )
        except Exception as exc:
            logger.warning("[ML] XGBoost erreur : %s", exc)

    # Tableau comparatif
    models_dict = {"RandomForest": rf_result}
    if xgb_result:
        models_dict["XGBoost"] = xgb_result

    comparison = build_comparison_table(models_dict)

    return {
        "rf": rf_result,
        "xgb": xgb_result,
        "target_distribution": y_aligned.value_counts().to_dict(),
        "comparison": comparison,
        "horizon": horizon,
        "X_aligned": X_aligned,
        "y_aligned": y_aligned,
    }

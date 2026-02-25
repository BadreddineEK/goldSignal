"""
backtester.py — Moteur de walk-forward validation pour GoldSignal.

Protocole strict : zero data leakage.
À chaque fold, le modèle est entraîné uniquement sur le passé
et prédit sur une fenêtre future qu'il n'a jamais vue.

Structure des folds Walk-Forward :
  |--- train ---|-- test --|
                 |--- train ---|-- test --|
                               |--- train ---|-- test --|

Métriques calculées :
  - RMSE, MAE, MAPE  (erreur sur prix ou log-return)
  - Directional Accuracy (DA%)  — métrique principale pour les signaux
  - Comparaison baseline random walk (DA% attendu ≈ 50% pour marché efficient)
"""

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pourcentage de fois où le signe de la prédiction est correct.

    Args:
        y_true: Valeurs réelles (log-return ou direction).
        y_pred: Valeurs prédites.

    Returns:
        Accuracy directionnelle entre 0 et 1.
    """
    if len(y_true) == 0:
        return 0.0
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(correct.mean())


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        epsilon: Évite la division par zéro.

    Returns:
        MAPE en fraction (ex: 0.02 = 2%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    name: str = "model") -> dict:
    """Calcule toutes les métriques pour un modèle.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        name: Nom du modèle pour le reporting.

    Returns:
        Dict avec clés : model, rmse, mae, mape_pct, da_pct, n_samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Masque NaN
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) == 0:
        return {"model": name, "rmse": np.nan, "mae": np.nan,
                "mape_pct": np.nan, "da_pct": np.nan, "n_samples": 0}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    mape_val = mape(yt, yp) * 100
    da = directional_accuracy(yt, yp) * 100

    return {
        "model": name,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape_pct": round(mape_val, 2),
        "da_pct": round(da, 2),
        "n_samples": int(len(yt)),
    }


# ---------------------------------------------------------------------------
# Baseline random walk
# ---------------------------------------------------------------------------

def random_walk_predictions(series: pd.Series) -> pd.Series:
    """Baseline naïf : la prédiction est la dernière valeur connue.

    Pour les log-returns, la prédiction est 0 (marché efficient).
    Pour les prix, la prédiction est le prix précédent.

    Args:
        series: Série temporelle à prédire.

    Returns:
        pd.Series des prédictions décalées d'un pas.
    """
    return series.shift(1)


def evaluate_random_walk(series: pd.Series) -> dict:
    """Évalue la baseline random walk sur toute la série.

    Args:
        series: Série de log-returns (ou prix).

    Returns:
        Dict de métriques (random walk).
    """
    y_pred = random_walk_predictions(series)
    valid = series.dropna().index.intersection(y_pred.dropna().index)
    return compute_metrics(
        series.loc[valid].values,
        y_pred.loc[valid].values,
        name="Random Walk",
    )


# ---------------------------------------------------------------------------
# Walk-Forward engine générique
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """Moteur de walk-forward validation.

    Découpe la série temporelle en N folds de type expanding window.
    Chaque fold entraîne sur [0 → train_end] et teste sur [train_end → test_end].

    Args:
        n_splits: Nombre de folds (défaut 5).
        min_train_size: Taille minimale de l'ensemble d'entraînement en observations.
        test_size: Nombre d'observations par fold de test.
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 500,
                 test_size: int = 60):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X: pd.DataFrame) -> list[tuple]:
        """Génère les indices train/test pour chaque fold.

        Args:
            X: DataFrame des features (index DatetimeIndex).

        Returns:
            Liste de tuples (train_idx, test_idx).
        """
        n = len(X)
        required = self.min_train_size + self.n_splits * self.test_size

        if n < required:
            logger.warning(
                "Données insuffisantes (%d obs). Minimum recommandé : %d.",
                n, required,
            )
            # Réduire le nombre de folds si nécessaire
            available_folds = max(1, (n - self.min_train_size) // self.test_size)
            self.n_splits = min(self.n_splits, available_folds)

        folds = []
        for k in range(self.n_splits):
            test_end = n - (self.n_splits - 1 - k) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start

            if train_end < self.min_train_size:
                continue

            folds.append((
                list(range(0, train_end)),
                list(range(test_start, test_end)),
            ))

        return folds

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_fn: Callable,
        predict_fn: Callable,
        model_name: str = "model",
    ) -> dict:
        """Exécute le walk-forward complet.

        Args:
            X: Features (DatetimeIndex aligné avec y).
            y: Cible (log-return ou direction).
            train_fn: Callable(X_train, y_train) → model.
            predict_fn: Callable(model, X_test) → np.ndarray de prédictions.
            model_name: Nom du modèle pour le reporting.

        Returns:
            Dict avec :
              - 'metrics': métriques agrégées sur tous les folds
              - 'metrics_per_fold': liste de métriques par fold
              - 'predictions': pd.Series des prédictions out-of-sample
              - 'actuals': pd.Series des vraies valeurs correspondantes
        """
        folds = self.split(X)
        if not folds:
            logger.error("Aucun fold généré. Dataset trop petit.")
            return {"metrics": {}, "metrics_per_fold": [], "predictions": pd.Series(), "actuals": pd.Series()}

        all_preds = []
        all_actuals = []
        metrics_per_fold = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            try:
                model = train_fn(X_train, y_train)
                preds = predict_fn(model, X_test)

                fold_metrics = compute_metrics(
                    y_test.values, preds,
                    name=f"{model_name}_fold{fold_idx + 1}",
                )
                metrics_per_fold.append(fold_metrics)

                all_preds.extend(zip(X.index[test_idx], preds))
                all_actuals.extend(zip(X.index[test_idx], y_test.values))

                logger.debug(
                    "Fold %d/%d — DA: %.1f%% | RMSE: %.6f",
                    fold_idx + 1, len(folds),
                    fold_metrics["da_pct"], fold_metrics["rmse"],
                )

            except Exception as exc:
                logger.warning("Erreur fold %d : %s", fold_idx + 1, exc)
                continue

        if not all_preds:
            return {"metrics": {}, "metrics_per_fold": [], "predictions": pd.Series(), "actuals": pd.Series()}

        pred_series = pd.Series(
            [v for _, v in all_preds],
            index=pd.DatetimeIndex([d for d, _ in all_preds]),
            name=f"{model_name}_pred",
        )
        actual_series = pd.Series(
            [v for _, v in all_actuals],
            index=pd.DatetimeIndex([d for d, _ in all_actuals]),
            name="actual",
        )

        # Métriques agrégées
        agg = compute_metrics(actual_series.values, pred_series.values, name=model_name)
        # DA par fold (pour graphique de stabilité temporelle)
        agg["da_folds"] = [round(m["da_pct"], 2) for m in metrics_per_fold]

        return {
            "metrics": agg,
            "metrics_per_fold": metrics_per_fold,
            "predictions": pred_series,
            "actuals": actual_series,
        }


# ---------------------------------------------------------------------------
# Formatage du tableau de comparaison
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Construit un tableau comparatif de tous les modèles.

    Args:
        results: Dict {model_name: walk-forward result dict}.

    Returns:
        DataFrame avec colonnes ['Modèle', 'DA%', 'RMSE', 'MAE', 'MAPE%', 'N'].
    """
    rows = []
    for name, res in results.items():
        m = res.get("metrics", {})
        if m:
            rows.append({
                "Modèle": name,
                "DA (%)": m.get("da_pct", np.nan),
                "RMSE": m.get("rmse", np.nan),
                "MAE": m.get("mae", np.nan),
                "MAPE (%)": m.get("mape_pct", np.nan),
                "N obs.": m.get("n_samples", 0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("DA (%)", ascending=False)
    return df

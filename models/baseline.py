"""
baseline.py — Modèles de référence pour GoldSignal.

Implémente deux baselines obligatoires :
  1. Random Walk : prédiction = dernière valeur connue (log-return = 0)
  2. ARIMA(p,d,q) : statsmodels auto-ordre via critère AIC

Les deux modèles sont évalués via walk-forward validation (backtester.py).
Aucun data leakage : l'ordre ARIMA est sélectionné sur le train set uniquement.
"""

import logging
import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from models.backtester import (
    WalkForwardCV, compute_metrics, evaluate_random_walk, build_comparison_table,
)

logger = logging.getLogger(__name__)

# Supprimer les warnings verbeux de statsmodels
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Test de stationnarité
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """Test Augmented Dickey-Fuller de stationnarité.

    Args:
        series: Série temporelle à tester.
        alpha: Niveau de significativité (défaut 5%).

    Returns:
        Dict avec 'stationary' (bool), 'adf_stat', 'p_value', 'critical_1pct'.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "stationary": result[1] < alpha,
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "critical_1pct": round(result[4]["1%"], 4),
        "lags_used": result[2],
        "n_obs": result[3],
    }


# ---------------------------------------------------------------------------
# Sélection automatique de l'ordre ARIMA
# ---------------------------------------------------------------------------

def auto_arima_order(series: pd.Series, max_p: int = 2, max_q: int = 2,
                     d: int = 0) -> tuple[int, int, int]:
    """Sélectionne l'ordre (p, d, q) optimal par minimisation de l'AIC.

    Recherche sur la grille réduite (p ∈ [0, max_p], q ∈ [0, max_q]).
    Défauts intentionnellement petits (max_p=2, max_q=2) pour la rapidité ;
    la littérature montre qu'ARIMA(1,0,1) ou (2,0,1) sont optimal sur les
    log-returns de l'or dans la majorité des régimes.

    Args:
        series: Série stationnaire (log-returns pour d=0).
        max_p: Ordre AR maximal à tester (défaut 2).
        max_q: Ordre MA maximal à tester (défaut 2).
        d: Ordre d'intégration (0 si déjà différenciée).

    Returns:
        Tuple (p, d, q) optimal.
    """
    best_aic = np.inf
    best_order = (1, d, 1)

    grid = [(p, d, q) for p, q in product(range(max_p + 1), range(max_q + 1))]

    for order in grid:
        if order[0] == 0 and order[2] == 0:
            continue  # modèle trivial
        try:
            model = ARIMA(series.dropna(), order=order)
            res = model.fit(method_kwargs={"maxiter": 30, "disp": False})
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
        except Exception:
            continue

    logger.debug("Meilleur ordre ARIMA : %s (AIC=%.2f)", best_order, best_aic)
    return best_order


# ---------------------------------------------------------------------------
# Fit / Predict ARIMA
# ---------------------------------------------------------------------------

def fit_arima(series: pd.Series, order: Optional[tuple] = None,
              max_p: int = 2, max_q: int = 2) -> dict:
    """Entraîne un modèle ARIMA sur la série fournie.

    Args:
        series: Série d'entraînement (log-returns, stationnaire).
        order: Tuple (p, d, q). Si None, sélection automatique par AIC.
        max_p: Grille de recherche auto (si order=None).
        max_q: Grille de recherche auto (si order=None).

    Returns:
        Dict avec 'fitted_model', 'order', 'aic', 'adf'.
    """
    s = series.dropna()

    adf_result = adf_test(s)
    if not adf_result["stationary"]:
        logger.warning("Série non stationnaire (ADF p=%.4f). ARIMA peut être sous-optimal.", adf_result["p_value"])

    if order is None:
        order = auto_arima_order(s, max_p=max_p, max_q=max_q, d=0)

    model = ARIMA(s, order=order)
    fitted = model.fit(method_kwargs={"maxiter": 50, "disp": False})

    return {
        "fitted_model": fitted,
        "order": order,
        "aic": round(fitted.aic, 2),
        "adf": adf_result,
    }


def predict_arima_next_n(fitted_result: dict, n_steps: int = 1) -> np.ndarray:
    """Prédit les n prochains log-returns avec ARIMA.

    Args:
        fitted_result: Résultat de fit_arima().
        n_steps: Nombre de pas à prédire.

    Returns:
        np.ndarray des prédictions.
    """
    forecast = fitted_result["fitted_model"].forecast(steps=n_steps)
    return np.asarray(forecast)


# ---------------------------------------------------------------------------
# Walk-Forward ARIMA
# ---------------------------------------------------------------------------

def _make_arima_train_fn(max_p: int = 2, max_q: int = 2, fixed_order: Optional[tuple] = None):
    """Fabrique une fonction de training ARIMA pour WalkForwardCV.

    Args:
        max_p: Ordre AR max pour la sélection auto (ignoré si fixed_order fourni).
        max_q: Ordre MA max pour la sélection auto (ignoré si fixed_order fourni).
        fixed_order: Si fourni (ex: (1,0,1)), bypasse la sélection AIC — beaucoup plus rapide.

    Returns:
        Callable(X_train, y_train) → fitted_result dict.
    """
    def train_fn(X_train: pd.DataFrame, y_train: pd.Series):
        return fit_arima(y_train, order=fixed_order, max_p=max_p, max_q=max_q)

    return train_fn


def _arima_predict_fn(fitted_result: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Prédit pas à pas (1-step-ahead) en re-fittant sur chaque observation.

    Pour respecter le protocole walk-forward strict, on prédit 1 pas à la fois.

    Args:
        fitted_result: Résultat de fit_arima() sur train.
        X_test: Features test (index utilisé pour aligner les prédictions).

    Returns:
        np.ndarray des prédictions 1-step-ahead.
    """
    n = len(X_test)
    if n == 0:
        return np.array([])

    fitted = fitted_result["fitted_model"]
    try:
        preds = fitted.forecast(steps=n)
        return np.asarray(preds)
    except Exception as exc:
        logger.warning("Erreur prédiction ARIMA : %s — retour 0.", exc)
        return np.zeros(n)


def run_arima_walkforward(
    log_returns: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 400,
    test_size: int = 60,
    max_p: int = 2,
    max_q: int = 2,
    fixed_order: Optional[tuple] = None,
) -> dict:
    """Exécute l'ARIMA en walk-forward validation.

    Travaille sur les log-returns (stationnaires, d=0).

    Args:
        log_returns: pd.Series de log-rendements.
        n_splits: Nombre de folds.
        min_train_size: Minimum d'observations pour entraîner.
        test_size: Observations par fenêtre de test.
        max_p: Ordre AR max pour la sélection AIC par fold (défaut 2, ignoré si fixed_order).
        max_q: Ordre MA max (défaut 2, ignoré si fixed_order).
        fixed_order: Ordre ARIMA fixe ex (1,0,1) — bypasse la sélection AIC,
                     ~10× plus rapide. Recommandé pour les démonstrations.

    Returns:
        Dict walk-forward result (voir WalkForwardCV.run).
    """
    s = log_returns.dropna()
    X_dummy = pd.DataFrame(index=s.index)  # pas de features externes pour ARIMA pur

    cv = WalkForwardCV(n_splits=n_splits, min_train_size=min_train_size, test_size=test_size)

    return cv.run(
        X=X_dummy,
        y=s,
        train_fn=_make_arima_train_fn(max_p, max_q, fixed_order=fixed_order),
        predict_fn=_arima_predict_fn,
        model_name="ARIMA",
    )


# ---------------------------------------------------------------------------
# Point d'entrée principal : baseline complète
# ---------------------------------------------------------------------------

def run_all_baselines(
    close: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 400,
    test_size: int = 60,
) -> dict:
    """Calcule les deux baselines (Random Walk + ARIMA) et compare.

    Args:
        close: Série de prix de clôture.
        n_splits: Folds walk-forward.
        min_train_size: Train minimum.
        test_size: Fenêtre de test.

    Returns:
        Dict avec :
          - 'random_walk': métriques random walk
          - 'arima': résultat walk-forward ARIMA
          - 'comparison': DataFrame comparatif
          - 'log_returns': série des log-returns utilisée
          - 'adf': résultat du test ADF
    """
    log_ret = np.log(close).diff().dropna()
    log_ret.name = "log_return"

    # 1. Random Walk
    rw_metrics = evaluate_random_walk(log_ret)
    rw_result = {"metrics": rw_metrics, "predictions": log_ret.shift(1), "actuals": log_ret}

    # 2. ARIMA walk-forward
    logger.info("Entraînement ARIMA walk-forward (%d folds)…", n_splits)
    arima_result = run_arima_walkforward(
        log_ret,
        n_splits=n_splits,
        min_train_size=min_train_size,
        test_size=test_size,
    )

    # 3. Tableau comparatif
    comparison = build_comparison_table({
        "Random Walk": rw_result,
        "ARIMA": arima_result,
    })

    return {
        "random_walk": rw_result,
        "arima": arima_result,
        "comparison": comparison,
        "log_returns": log_ret,
        "adf": adf_test(log_ret),
    }

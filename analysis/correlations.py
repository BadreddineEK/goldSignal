"""
correlations.py — Analyse des corrélations pour GoldSignal.

Calcule les corrélations rolling entre le cours de l'or et les autres actifs,
ainsi que les heatmaps de corrélation mensuelle.
"""

import numpy as np
import pandas as pd


def rolling_correlation(s1: pd.Series, s2: pd.Series,
                         window: int = 60) -> pd.Series:
    """Corrélation rolling de Pearson entre deux séries.

    Args:
        s1: Première série (ex: log-rendements or).
        s2: Deuxième série (ex: log-rendements DXY).
        window: Fenêtre glissante en jours (défaut 60).

    Returns:
        pd.Series corrélation entre -1 et 1.
    """
    aligned = pd.concat([s1, s2], axis=1).dropna()
    r1, r2 = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return r1.rolling(window).corr(r2)


def compute_log_returns(s: pd.Series) -> pd.Series:
    """Log-rendements journaliers.

    Args:
        s: Série de prix.

    Returns:
        pd.Series log-rendements.
    """
    return np.log(s).diff()


def build_correlation_matrix(prices: dict[str, pd.Series],
                               use_log_returns: bool = True) -> pd.DataFrame:
    """Calcule la matrice de corrélation entre plusieurs actifs.

    Args:
        prices: Dict {nom: pd.Series de prix de clôture}.
        use_log_returns: Si True, corrèle les log-rendements (recommandé).

    Returns:
        DataFrame de corrélation (NxN).
    """
    if use_log_returns:
        data = {name: compute_log_returns(s) for name, s in prices.items()}
    else:
        data = prices

    df = pd.concat(data, axis=1).dropna()
    return df.corr()


def build_monthly_correlation_heatmap(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Heatmap de corrélation basée sur des rendements mensuels.

    Args:
        prices: Dict {nom: pd.Series de prix de clôture quotidiens}.

    Returns:
        DataFrame de corrélation mensuelle.
    """
    monthly = {}
    for name, s in prices.items():
        s_monthly = s.resample("ME").last()
        monthly[name] = compute_log_returns(s_monthly)

    df = pd.concat(monthly, axis=1).dropna()
    return df.corr()


def taux_reels_vs_or(close_xau: pd.Series,
                      taux_reels: pd.Series) -> pd.DataFrame:
    """Aligne cours de l'or et taux réels pour scatter plot + corr rolling.

    Args:
        close_xau: Série de prix de l'or.
        taux_reels: Série taux réels TIPS10 (FRED DFII10, en %).

    Returns:
        DataFrame aligné avec colonnes ['xau', 'taux_reels', 'corr_rolling_60'].
    """
    df = pd.concat(
        [close_xau.rename("xau"), taux_reels.rename("taux_reels")],
        axis=1,
    ).ffill(limit=5).dropna()

    log_ret_xau = np.log(df["xau"]).diff()
    diff_taux = df["taux_reels"].diff()

    corr = log_ret_xau.rolling(60).corr(diff_taux)

    df["corr_rolling_60"] = corr
    return df

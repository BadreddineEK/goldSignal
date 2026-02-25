"""
processor.py — Nettoyage, transformation et feature engineering pour GoldSignal.

Ce module prend en entrée des DataFrames bruts (yfinance / FRED) et produit
des DataFrames propres enrichis de features pour l'analyse et les modèles ML.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TROY_OZ_G = 31.1035   # grammes dans 1 troy once


# ---------------------------------------------------------------------------
# Conversion USD → EUR
# ---------------------------------------------------------------------------

def convert_usd_to_eur(df_usd: pd.DataFrame, df_eurusd: pd.DataFrame,
                        cols: list[str] = None) -> pd.DataFrame:
    """Convertit des colonnes prix USD en EUR en divisant par EUR/USD.

    Args:
        df_usd: DataFrame avec prix en USD.
        df_eurusd: DataFrame EUR/USD (colonne 'close').
        cols: Colonnes à convertir. Par défaut ['open', 'high', 'low', 'close'].

    Returns:
        DataFrame avec les colonnes converties en EUR.
    """
    if cols is None:
        cols = [c for c in ["open", "high", "low", "close"] if c in df_usd.columns]

    df = df_usd.copy()
    eur = df_eurusd["close"].reindex(df.index, method="ffill")

    for col in cols:
        if col in df.columns:
            df[col] = df[col] / eur

    return df


def ohlcv_usd_oz_to_eur_g(df_usd: pd.DataFrame, df_eurusd: pd.DataFrame) -> pd.DataFrame:
    """Convertit un OHLCV en USD/oz troy en EUR/g fin.

    Divise par EUR/USD puis par 31.1035 g/troy oz.

    Args:
        df_usd: OHLCV en USD/oz (ex: GC=F, SI=F).
        df_eurusd: OHLCV EUR/USD (ex: EURUSD=X).

    Returns:
        DataFrame en EUR/g, mêmes colonnes OHLCV.
    """
    df_eur = convert_usd_to_eur(df_usd, df_eurusd)
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df_eur.columns]
    for col in price_cols:
        df_eur[col] = df_eur[col] / TROY_OZ_G
    return df_eur


# ---------------------------------------------------------------------------
# Nettoyage & alignement
# ---------------------------------------------------------------------------

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame OHLCV.

    - Supprime les lignes avec close <= 0
    - Forward-fill les valeurs manquantes (max 5 jours)
    - Supprime les lignes sans close valide restantes

    Args:
        df: DataFrame OHLCV brut.

    Returns:
        DataFrame nettoyé.
    """
    df = df.copy()
    if "close" in df.columns:
        df.loc[df["close"] <= 0, "close"] = np.nan
    df = df.ffill(limit=5)
    df = df.dropna(subset=["close"])
    return df


def align_series(*series: pd.Series, method: str = "inner") -> pd.DataFrame:
    """Aligne plusieurs séries temporelles sur un index commun.

    Args:
        *series: Séries pd.Series à aligner.
        method: 'inner' (intersection) ou 'outer' (union avec NaN).

    Returns:
        DataFrame avec une colonne par série.
    """
    df = pd.concat(series, axis=1, join=method)
    df = df.ffill(limit=5)
    return df


# ---------------------------------------------------------------------------
# Calcul ratio or / argent
# ---------------------------------------------------------------------------

def compute_gold_silver_ratio(df_xau_eur: pd.DataFrame,
                               df_xag_eur: pd.DataFrame) -> pd.Series:
    """Calcule le ratio or/argent (en prix, pas en g).

    Ratio = cours or (USD/oz) / cours argent (USD/oz).
    Utilise la colonne 'close'.

    Args:
        df_xau_eur: OHLCV or (quelque monnaie).
        df_xag_eur: OHLCV argent (même monnaie).

    Returns:
        pd.Series du ratio, index DatetimeIndex.
    """
    xau = df_xau_eur["close"].rename("xau")
    xag = df_xag_eur["close"].rename("xag")
    aligned = pd.concat([xau, xag], axis=1).dropna()
    ratio = aligned["xau"] / aligned["xag"]
    ratio.name = "ratio_or_argent"
    return ratio


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_log_returns(s: pd.Series, lags: list[int] = None) -> pd.DataFrame:
    """Calcule les log-rendements à plusieurs horizons.

    Args:
        s: Série de prix (close).
        lags: Liste de décalages (jours). Défaut : [1, 2, 5].

    Returns:
        DataFrame avec colonnes 'log_return_{lag}'.
    """
    if lags is None:
        lags = [1, 2, 5]
    result = {}
    log_s = np.log(s)
    for lag in lags:
        result[f"log_return_{lag}"] = log_s.diff(lag)
    return pd.DataFrame(result, index=s.index)


def compute_sma_ratios(close: pd.Series,
                        windows: list[int] = None) -> pd.DataFrame:
    """Calcule le ratio prix/SMA pour différentes fenêtres.

    Args:
        close: Série de cours de clôture.
        windows: Fenêtres en jours. Défaut : [20, 50, 200].

    Returns:
        DataFrame avec colonnes 'prix_sur_sma{w}'.
    """
    if windows is None:
        windows = [20, 50, 200]
    result = {}
    for w in windows:
        sma = close.rolling(w, min_periods=w // 2).mean()
        result[f"prix_sur_sma{w}"] = close / sma
    return pd.DataFrame(result, index=close.index)


def build_feature_matrix(
    df_xau: pd.DataFrame,
    df_xag: pd.DataFrame,
    s_taux_reels: pd.Series,
    df_dxy: pd.DataFrame,
    df_vix: pd.DataFrame,
    rsi_series: pd.Series,
    atr_series: pd.Series,
) -> pd.DataFrame:
    """Assemble la matrice de features pour les modèles ML.

    Features incluses :
      - Log-rendements J-1, J-2, J-5 (or)
      - Prix/SMA20, Prix/SMA50
      - RSI(14), ATR(14)
      - Taux réels lag-1, lag-5
      - DXY rendement lag-1
      - VIX niveau
      - Ratio or/argent
      - Mois de l'année (1-12, saisonnalité)

    Args:
        df_xau: OHLCV or.
        df_xag: OHLCV argent.
        s_taux_reels: Série FRED taux réels (%).
        df_dxy: OHLCV DXY.
        df_vix: OHLCV VIX.
        rsi_series: Série RSI(14) précalculée.
        atr_series: Série ATR(14) précalculée.

    Returns:
        DataFrame de features, index DatetimeIndex, sans NaN.
    """
    close_xau = df_xau["close"]

    # Log-rendements
    log_rets = compute_log_returns(close_xau, [1, 2, 5])

    # SMA ratios
    sma_ratios = compute_sma_ratios(close_xau, [20, 50])

    # Taux réels avec lags
    taux_aligned = s_taux_reels.reindex(close_xau.index, method="ffill")
    taux_df = pd.DataFrame({
        "taux_reels_lag1": taux_aligned.shift(1),
        "taux_reels_lag5": taux_aligned.shift(5),
    }, index=close_xau.index)

    # DXY rendement lag-1
    dxy_ret = np.log(df_dxy["close"]).diff(1).shift(1)
    dxy_ret.name = "dxy_return_lag1"
    dxy_aligned = dxy_ret.reindex(close_xau.index, method="ffill")

    # VIX
    vix_aligned = df_vix["close"].reindex(close_xau.index, method="ffill")
    vix_aligned.name = "vix_niveau"

    # Ratio or/argent
    ratio = compute_gold_silver_ratio(df_xau, df_xag)
    ratio_aligned = ratio.reindex(close_xau.index, method="ffill")

    # RSI & ATR
    rsi_aligned = rsi_series.reindex(close_xau.index, method="ffill")
    rsi_aligned.name = "rsi_14"
    atr_aligned = atr_series.reindex(close_xau.index, method="ffill")
    atr_aligned.name = "atr_14"

    # Mois
    mois = pd.Series(close_xau.index.month, index=close_xau.index, name="mois")

    # Assemblage
    features = pd.concat([
        log_rets,
        sma_ratios,
        taux_df,
        dxy_aligned,
        vix_aligned,
        ratio_aligned,
        rsi_aligned,
        atr_aligned,
        mois,
    ], axis=1)

    features = features.dropna()
    return features


# ---------------------------------------------------------------------------
# Cible : direction (haussier / neutre / baissier)
# ---------------------------------------------------------------------------

def compute_target_direction(close: pd.Series, horizon: int = 5,
                              seuil_pct: float = 0.5) -> pd.Series:
    """Crée la variable cible de direction à H jours.

    - 1  : rendement futur > +seuil_pct %  (haussier)
    - 0  : rendement futur dans [-seuil ; +seuil]  (neutre)
    - -1 : rendement futur < -seuil_pct %  (baissier)

    Args:
        close: Série de cours de clôture.
        horizon: Nombre de jours dans le futur.
        seuil_pct: Seuil en % (ex: 0.5 = 0.5%).

    Returns:
        pd.Series de {-1, 0, 1} alignée sur close.
    """
    future_ret = close.shift(-horizon) / close - 1  # rendement en fraction
    seuil = seuil_pct / 100.0
    direction = future_ret.apply(
        lambda r: 1 if r > seuil else (-1 if r < -seuil else 0)
    )
    direction.name = f"direction_{horizon}j"
    return direction

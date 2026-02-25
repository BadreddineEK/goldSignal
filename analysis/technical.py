"""
technical.py — Indicateurs techniques pour GoldSignal.

Tous les indicateurs opèrent sur des pd.Series de prix de clôture (ou OHLC).
Aucune dépendance externe TA-lib : calculs natifs numpy/pandas.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SMA / EMA
# ---------------------------------------------------------------------------

def sma(close: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple.

    Args:
        close: Série de prix de clôture.
        window: Fenêtre en jours.

    Returns:
        pd.Series SMA, même index que close.
    """
    return close.rolling(window, min_periods=window // 2).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile exponentielle.

    Args:
        close: Série de prix de clôture.
        window: Fenêtre (span).

    Returns:
        pd.Series EMA.
    """
    return close.ewm(span=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) de Wilder.

    Args:
        close: Série de prix de clôture.
        period: Période RSI (défaut 14).

    Returns:
        pd.Series RSI entre 0 et 100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series.name = f"rsi_{period}"
    return rsi_series


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(close: pd.Series, window: int = 20,
                    n_sigma: float = 2.0) -> pd.DataFrame:
    """Bandes de Bollinger.

    Args:
        close: Série de clôture.
        window: Fenêtre SMA centrale (défaut 20).
        n_sigma: Nombre d'écarts-types (défaut 2.0).

    Returns:
        DataFrame avec colonnes ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct'].
          bb_pct = position du prix dans la bande (0 = lower, 1 = upper).
    """
    middle = sma(close, window)
    std = close.rolling(window, min_periods=window // 2).std()
    upper = middle + n_sigma * std
    lower = middle - n_sigma * std
    width = (upper - lower) / middle
    pct = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct": pct,
    }, index=close.index)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD : Moving Average Convergence/Divergence.

    Args:
        close: Série de clôture.
        fast: Fenêtre EMA rapide (défaut 12).
        slow: Fenêtre EMA lente (défaut 26).
        signal: Fenêtre EMA de la ligne signal (défaut 9).

    Returns:
        DataFrame avec colonnes ['macd_line', 'macd_signal', 'macd_hist'].
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=close.index)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (mesure de volatilité).

    Args:
        df_ohlc: DataFrame avec colonnes 'high', 'low', 'close'.
        period: Période ATR (défaut 14).

    Returns:
        pd.Series ATR.
    """
    high = df_ohlc["high"]
    low = df_ohlc["low"]
    prev_close = df_ohlc["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr_series.name = f"atr_{period}"
    return atr_series


# ---------------------------------------------------------------------------
# Enrichissement complet d'un OHLCV
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame,
                        sma_windows: list[int] = None,
                        rsi_period: int = 14,
                        bb_window: int = 20,
                        bb_sigma: float = 2.0,
                        macd_fast: int = 12,
                        macd_slow: int = 26,
                        macd_signal: int = 9,
                        atr_period: int = 14) -> pd.DataFrame:
    """Ajoute tous les indicateurs techniques à un DataFrame OHLCV.

    Args:
        df: DataFrame avec au moins 'close' (et 'high', 'low' pour ATR).
        sma_windows: Fenêtres SMA à ajouter (défaut [20, 50, 200]).
        rsi_period: Période RSI.
        bb_window: Fenêtre Bollinger.
        bb_sigma: Sigma Bollinger.
        macd_fast/slow/signal: Paramètres MACD.
        atr_period: Période ATR.

    Returns:
        DataFrame enrichi.
    """
    if sma_windows is None:
        sma_windows = [20, 50, 200]

    result = df.copy()
    close = result["close"]

    # SMA
    for w in sma_windows:
        result[f"sma{w}"] = sma(close, w)

    # RSI
    result[f"rsi_{rsi_period}"] = rsi(close, rsi_period)

    # Bollinger
    bb = bollinger_bands(close, bb_window, bb_sigma)
    result = pd.concat([result, bb], axis=1)

    # MACD
    macd_df = macd(close, macd_fast, macd_slow, macd_signal)
    result = pd.concat([result, macd_df], axis=1)

    # ATR (nécessite high/low)
    if "high" in result.columns and "low" in result.columns:
        result[f"atr_{atr_period}"] = atr(result, atr_period)

    return result

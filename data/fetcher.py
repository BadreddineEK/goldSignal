"""
fetcher.py — Récupération des données de marché pour GoldSignal.

Sources :
  - yfinance   : or, argent, DXY, S&P500, VIX, EUR/USD, TNX, WTI
  - FRED API   : taux réels TIPS10 (DFII10), CPI (CPIAUCSL)

Stratégie de cache :
  Les données sont stockées dans SQLite (price_cache / macro_cache).
  Si la dernière date en cache est aujourd'hui (ou hier hors jours ouvrés),
  on ne refetch pas. Sinon on récupère uniquement la période manquante.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd
import yfinance as yf

from data.database import (
    get_cached_prices,
    get_cache_last_date,
    insert_prices_bulk,
    get_cached_macro,
    insert_macro_bulk,
    get_config,
)

logger = logging.getLogger(__name__)

# Date de démarrage par défaut pour l'historique initial
_HISTORY_START = "2000-01-01"


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return date.today().isoformat()


def _yesterday_str() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _cache_is_fresh(last_date: Optional[str], ttl_heures: int = 24) -> bool:
    """Retourne True si le cache est suffisamment récent.

    On considère le cache frais si la dernière date stockée est :
      - aujourd'hui OU hier (pour les marchés fermés le weekend).

    Args:
        last_date: Dernière date en cache ('YYYY-MM-DD') ou None.
        ttl_heures: Non utilisé directement (logique jour ouvré).

    Returns:
        True si frais, False sinon.
    """
    if last_date is None:
        return False
    today = date.today()
    last = date.fromisoformat(last_date)
    delta = (today - last).days
    # Frais si delta <= 3 jours (couvre weekend + jour férié)
    return delta <= 3


# ---------------------------------------------------------------------------
# yfinance — prix OHLCV
# ---------------------------------------------------------------------------

def fetch_ticker(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    """Récupère les données OHLCV pour un ticker yfinance.

    Lit d'abord le cache SQLite. Si incomplet ou expiré, télécharge
    uniquement la période manquante puis met à jour le cache.

    Args:
        ticker: Ticker yfinance (ex: 'GC=F', '^VIX').
        force_refresh: Si True, ignore le cache et re-télécharge tout.

    Returns:
        DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
        indexé par date (DatetimeIndex, tz-naive, fréquence journalière).
    """
    cfg = get_config("api") or {}
    ttl = cfg.get("cache_ttl_heures", 24)

    last_cached = get_cache_last_date(ticker)

    if not force_refresh and _cache_is_fresh(last_cached, ttl):
        logger.debug("[%s] Cache frais, lecture SQLite.", ticker)
        return _load_from_cache(ticker)

    # Détermine la période à télécharger
    if last_cached and not force_refresh:
        start = last_cached  # incrémental
    else:
        start = _HISTORY_START

    end = _today_str()

    logger.info("[%s] Téléchargement yfinance %s → %s", ticker, start, end)
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.warning("[%s] Erreur yfinance : %s — lecture cache.", ticker, exc)
        return _load_from_cache(ticker)

    if raw.empty:
        logger.warning("[%s] Aucune donnée retournée par yfinance.", ticker)
        return _load_from_cache(ticker)

    # Aplatir colonnes MultiIndex si présentes (yfinance ≥ 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    # Sauvegarde en cache
    rows = [
        {
            "date_str": idx.date().isoformat(),
            "open": float(row.get("open", 0) or 0),
            "high": float(row.get("high", 0) or 0),
            "low": float(row.get("low", 0) or 0),
            "close": float(row.get("close", 0) or 0),
            "volume": float(row.get("volume", 0) or 0),
        }
        for idx, row in raw.iterrows()
    ]
    insert_prices_bulk(ticker, rows)

    return _load_from_cache(ticker)


def _load_from_cache(ticker: str) -> pd.DataFrame:
    """Charge les données d'un ticker depuis le cache SQLite.

    Args:
        ticker: Ticker yfinance.

    Returns:
        DataFrame OHLCV indexé par DatetimeIndex.
    """
    rows = get_cached_prices(ticker)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date_str"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
    })
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# Spots or/argent en EUR
# ---------------------------------------------------------------------------

def get_spot_xau_eur(force_refresh: bool = False) -> Optional[float]:
    """Retourne le cours spot actuel de l'or en €/g fin.

    Convertit GC=F (USD/oz troy) → EUR/oz → EUR/g.

    Args:
        force_refresh: Forcer le rechargement depuis yfinance.

    Returns:
        Cours en €/g fin ou None si données indisponibles.
    """
    df_xau = fetch_ticker("GC=F", force_refresh=force_refresh)
    df_eur = fetch_ticker("EURUSD=X", force_refresh=force_refresh)

    if df_xau.empty or df_eur.empty:
        return None

    xau_usd_oz = float(df_xau["close"].iloc[-1])   # USD par troy oz (31.1035 g)
    eurusd = float(df_eur["close"].iloc[-1])
    xau_eur_oz = xau_usd_oz / eurusd
    xau_eur_g = xau_eur_oz / 31.1035
    return round(xau_eur_g, 4)


def get_spot_xag_eur(force_refresh: bool = False) -> Optional[float]:
    """Retourne le cours spot actuel de l'argent en €/g fin.

    Convertit SI=F (USD/oz troy) → EUR/oz → EUR/g.

    Args:
        force_refresh: Forcer le rechargement depuis yfinance.

    Returns:
        Cours en €/g fin ou None si données indisponibles.
    """
    df_xag = fetch_ticker("SI=F", force_refresh=force_refresh)
    df_eur = fetch_ticker("EURUSD=X", force_refresh=force_refresh)

    if df_xag.empty or df_eur.empty:
        return None

    xag_usd_oz = float(df_xag["close"].iloc[-1])
    eurusd = float(df_eur["close"].iloc[-1])
    xag_eur_oz = xag_usd_oz / eurusd
    xag_eur_g = xag_eur_oz / 31.1035
    return round(xag_eur_g, 4)


def get_spot_xau_usd(force_refresh: bool = False) -> Optional[float]:
    """Retourne le cours spot de l'or en USD/oz troy."""
    df = fetch_ticker("GC=F", force_refresh=force_refresh)
    if df.empty:
        return None
    return round(float(df["close"].iloc[-1]), 2)


def get_spot_xag_usd(force_refresh: bool = False) -> Optional[float]:
    """Retourne le cours spot de l'argent en USD/oz troy."""
    df = fetch_ticker("SI=F", force_refresh=force_refresh)
    if df.empty:
        return None
    return round(float(df["close"].iloc[-1]), 2)


# ---------------------------------------------------------------------------
# FRED API — taux réels et CPI
# ---------------------------------------------------------------------------

def _get_fred_key() -> Optional[str]:
    """Retourne la clé API FRED.

    Priorité :
      1. st.secrets["fred_api_key"]  (Streamlit Community Cloud)
      2. config SQLite (page Config de l'app)
    """
    # 1. Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "fred_api_key" in st.secrets:
            return str(st.secrets["fred_api_key"])
    except Exception:
        pass
    # 2. Config SQLite locale
    cfg = get_config("api") or {}
    return cfg.get("fred_api_key") or None


def fetch_fred_series(serie_id: str, force_refresh: bool = False) -> pd.Series:
    """Télécharge une série FRED et la met en cache.

    Nécessite une clé API FRED dans la config (config.api.fred_api_key).
    Si la clé est absente, tente sans authentification via pandas-datareader.

    Args:
        serie_id: Identifiant FRED (ex: 'DFII10', 'CPIAUCSL').
        force_refresh: Forcer le rechargement.

    Returns:
        pd.Series indexé par DatetimeIndex, ou Series vide si erreur.
    """
    cached = get_cached_macro(serie_id)

    if cached and not force_refresh:
        last_date = max(r["date_str"] for r in cached)
        if _cache_is_fresh(last_date):
            return _fred_from_cache(serie_id)

    fred_key = _get_fred_key()

    try:
        if fred_key:
            from fredapi import Fred  # type: ignore
            fred = Fred(api_key=fred_key)
            data = fred.get_series(serie_id, observation_start=_HISTORY_START)
        else:
            # Fallback sans clé via pandas-datareader
            import pandas_datareader.data as web  # type: ignore
            data = web.DataReader(serie_id, "fred", _HISTORY_START)
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]

        data = data.dropna()
        data.index = pd.to_datetime(data.index).tz_localize(None)

        rows = [
            {"date_str": idx.date().isoformat(), "value": float(val)}
            for idx, val in data.items()
        ]
        insert_macro_bulk(serie_id, rows)
        logger.info("[FRED:%s] %d observations mises en cache.", serie_id, len(rows))

    except Exception as exc:
        logger.warning("[FRED:%s] Erreur récupération : %s", serie_id, exc)

    return _fred_from_cache(serie_id)


def _fred_from_cache(serie_id: str) -> pd.Series:
    """Charge une série FRED depuis le cache SQLite.

    Args:
        serie_id: Identifiant FRED.

    Returns:
        pd.Series indexé par DatetimeIndex.
    """
    rows = get_cached_macro(serie_id)
    if not rows:
        return pd.Series(dtype=float, name=serie_id)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date_str"])
    s = df.set_index("date")["value"].sort_index()
    s.name = serie_id
    return s


def get_taux_reels(force_refresh: bool = False) -> pd.Series:
    """Retourne la série TIPS 10 ans (taux réels US) depuis FRED.

    Args:
        force_refresh: Forcer le rechargement.

    Returns:
        pd.Series en %, index DatetimeIndex.
    """
    return fetch_fred_series("DFII10", force_refresh=force_refresh)


def get_cpi(force_refresh: bool = False) -> pd.Series:
    """Retourne la série CPI US (indice des prix à la consommation) depuis FRED.

    Args:
        force_refresh: Forcer le rechargement.

    Returns:
        pd.Series index DatetimeIndex.
    """
    return fetch_fred_series("CPIAUCSL", force_refresh=force_refresh)


# ---------------------------------------------------------------------------
# Batch fetch (toutes les séries en une seule fois)
# ---------------------------------------------------------------------------

def fetch_all_market_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Télécharge toutes les séries de marché yfinance configurées.

    Args:
        force_refresh: Forcer le rechargement de toutes les séries.

    Returns:
        Dict {ticker: DataFrame OHLCV} pour chaque ticker configuré.
    """
    cfg = get_config("tickers_yfinance") or {}
    tickers = list(cfg.values()) if cfg else [
        "GC=F", "SI=F", "DX-Y.NYB", "^GSPC", "^TNX",
        "EURUSD=X", "^VIX", "CL=F",
    ]

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_ticker(ticker, force_refresh=force_refresh)
        except Exception as exc:
            logger.warning("Erreur fetch %s : %s", ticker, exc)
            result[ticker] = pd.DataFrame()

    return result

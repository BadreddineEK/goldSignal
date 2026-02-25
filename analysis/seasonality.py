"""
seasonality.py — Analyse de saisonnalité mensuelle pour GoldSignal.

Calcule les rendements moyens et médians par mois et par trimestre
depuis 2000, pour identifier les patterns saisonniers historiques de l'or
et de l'argent.
"""

import numpy as np
import pandas as pd


MOIS_FR = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Aoû",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc",
}

TRIMESTRES_FR = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}


def monthly_returns(close: pd.Series) -> pd.DataFrame:
    """Calcule les rendements mensuels depuis 2000.

    Args:
        close: Série de prix de clôture quotidiens (DatetimeIndex).

    Returns:
        DataFrame avec colonnes ['annee', 'mois', 'rendement_pct'].
    """
    since_2000 = close[close.index >= "2000-01-01"]
    monthly = since_2000.resample("ME").last()
    ret = monthly.pct_change() * 100
    ret = ret.dropna()

    df = pd.DataFrame({
        "annee": ret.index.year,
        "mois": ret.index.month,
        "rendement_pct": ret.values,
    })
    return df


def seasonality_by_month(close: pd.Series) -> pd.DataFrame:
    """Statistiques de saisonnalité par mois (rendement moyen/médian/std).

    Args:
        close: Série de prix de clôture quotidiens.

    Returns:
        DataFrame indexé par mois (1-12) avec colonnes :
        ['mois_nom', 'rendement_moyen_pct', 'rendement_median_pct',
         'rendement_std_pct', 'nb_annees'].
    """
    df = monthly_returns(close)
    grouped = df.groupby("mois")["rendement_pct"].agg(
        rendement_moyen_pct="mean",
        rendement_median_pct="median",
        rendement_std_pct="std",
        nb_annees="count",
    ).reset_index()
    grouped["mois_nom"] = grouped["mois"].map(MOIS_FR)
    return grouped.set_index("mois")


def seasonality_by_quarter(close: pd.Series) -> pd.DataFrame:
    """Statistiques de saisonnalité par trimestre.

    Args:
        close: Série de prix de clôture quotidiens.

    Returns:
        DataFrame indexé par trimestre (1-4).
    """
    df = monthly_returns(close)
    df["trimestre"] = ((df["mois"] - 1) // 3) + 1
    grouped = df.groupby("trimestre")["rendement_pct"].agg(
        rendement_moyen_pct="mean",
        rendement_median_pct="median",
        rendement_std_pct="std",
        nb_trimestres="count",
    ).reset_index()
    grouped["trimestre_nom"] = grouped["trimestre"].map(TRIMESTRES_FR)
    return grouped.set_index("trimestre")


def best_worst_months(close: pd.Series, top_n: int = 3) -> dict:
    """Retourne les meilleurs et pires mois historiques.

    Args:
        close: Série de prix de clôture quotidiens.
        top_n: Nombre de mois à retourner dans chaque catégorie.

    Returns:
        Dict {'meilleurs': list[mois_nom], 'pires': list[mois_nom]}.
    """
    stats = seasonality_by_month(close)
    sorted_asc = stats.sort_values("rendement_moyen_pct")
    pires = [MOIS_FR[m] for m in sorted_asc.head(top_n).index.tolist()]
    meilleurs = [MOIS_FR[m] for m in sorted_asc.tail(top_n).index.tolist()]
    return {"meilleurs": list(reversed(meilleurs)), "pires": pires}

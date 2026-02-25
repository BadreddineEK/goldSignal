"""
macro_score.py — Score macro composite pour GoldSignal.

Calcule un score synthétique sur 5 axes pour évaluer si le contexte
macroéconomique est favorable à l'achat de métaux précieux.

Score_Macro = f(taux_réels, DXY_trend, VIX_level, RSI_or, ratio_or_argent_percentile)

Chaque axe est noté de -1 (défavorable) à +1 (favorable), le score
composite est la somme (borné à [-5, +5]).

Verdict :
  ✅ Favorable  : score >= seuil_achat (défaut 3)
  ⚠️  Neutre    : score entre 0 et seuil_achat
  ❌ Défavorable : score < 0
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Sous-scores individuels
# ---------------------------------------------------------------------------

def score_taux_reels(taux_reel_dernier: float) -> float:
    """Score basé sur le niveau des taux réels US.

    Taux réels négatifs → favorable pour l'or (+1).
    Taux élevés (> 2 %) → défavorable (-1).

    Args:
        taux_reel_dernier: Dernier taux réel TIPS10 en %.

    Returns:
        Score dans {-1, 0, +1}.
    """
    if taux_reel_dernier < 0:
        return 1.0
    elif taux_reel_dernier < 1.0:
        return 0.0
    else:
        return -1.0


def score_dxy_trend(close_dxy: pd.Series, window: int = 20) -> float:
    """Score basé sur la tendance récente du Dollar Index.

    DXY en baisse → favorable pour l'or (+1).
    DXY en hausse → défavorable (-1).

    Args:
        close_dxy: Série de prix DXY.
        window: Fenêtre pour calculer la tendance (défaut 20 jours).

    Returns:
        Score dans {-1, 0, +1}.
    """
    if len(close_dxy) < window:
        return 0.0
    recent = close_dxy.iloc[-window:]
    trend = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100  # rendement %
    if trend < -1.0:
        return 1.0
    elif trend > 1.0:
        return -1.0
    else:
        return 0.0


def score_vix(vix_level: float, fear_threshold: float = 30.0,
              greed_threshold: float = 15.0) -> float:
    """Score basé sur le niveau du VIX (indice de volatilité/peur).

    VIX élevé (peur) → souvent favorable pour l'or en refuge (+1).
    VIX très bas (cupidité) → neutre à négatif (0).

    Args:
        vix_level: Niveau actuel du VIX.
        fear_threshold: Seuil de peur (défaut 30).
        greed_threshold: Seuil de cupidité (défaut 15).

    Returns:
        Score dans {-1, 0, +1}.
    """
    if vix_level >= fear_threshold:
        return 1.0
    elif vix_level <= greed_threshold:
        return 0.0
    else:
        return 0.5


def score_rsi_or(rsi_value: float) -> float:
    """Score basé sur le RSI(14) de l'or.

    RSI survendu (< 35) → favorable (potentiel rebond) → +1.
    RSI suracheté (> 70) → défavorable → -1.
    Zone neutre entre 35 et 70 → 0.

    Args:
        rsi_value: Valeur RSI entre 0 et 100.

    Returns:
        Score dans {-1, -0.5, 0, +1}.
    """
    if rsi_value < 35:
        return 1.0
    elif rsi_value < 45:
        return 0.5
    elif rsi_value > 70:
        return -1.0
    elif rsi_value > 60:
        return -0.5
    else:
        return 0.0


def score_ratio_or_argent(ratio: float, ratio_series: pd.Series,
                           window_years: int = 20) -> float:
    """Score basé sur le percentile historique du ratio or/argent.

    Ratio élevé (argent sous-évalué relatif) → potentiel achat argent,
    mais aussi signal de stress/incertitude → neutre à légèrement positif.
    Ratio au plus bas (> 80e percentile historique) → signal de surchauffe or.

    Args:
        ratio: Ratio actuel or/argent.
        ratio_series: Série historique du ratio.
        window_years: Nombre d'années pour le calcul de percentile (défaut 20).

    Returns:
        Score entre -1 et +1.
    """
    if len(ratio_series) < 252:  # moins d'1 an
        return 0.0

    window_days = min(window_years * 252, len(ratio_series))
    hist = ratio_series.iloc[-window_days:]
    pct = float((hist < ratio).mean() * 100)  # percentile du ratio actuel

    if pct >= 85:
        # Ratio historiquement très élevé → or cher vs argent, signal contrarian négatif
        return -0.5
    elif pct >= 70:
        return -0.25
    elif pct <= 25:
        # Ratio faible → or relativement peu cher vs argent
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Score composite
# ---------------------------------------------------------------------------

def compute_macro_score(
    taux_reel_dernier: float,
    close_dxy: pd.Series,
    vix_level: float,
    rsi_or: float,
    ratio_or_argent: float,
    ratio_series: pd.Series,
    cfg: dict = None,
) -> dict:
    """Calcule le score macro composite GoldSignal sur 5 axes.

    Args:
        taux_reel_dernier: Dernier taux réel TIPS10 (%).
        close_dxy: Série historique DXY.
        vix_level: Niveau VIX actuel.
        rsi_or: RSI(14) de l'or actuel.
        ratio_or_argent: Ratio or/argent actuel.
        ratio_series: Série historique du ratio or/argent.
        cfg: Config macro (issu de default_config.json section 'macro').

    Returns:
        Dict avec :
          - 'score_total': float entre -5 et +5
          - 'verdict': 'favorable' | 'neutre' | 'défavorable'
          - 'details': dict des sous-scores par axe
    """
    if cfg is None:
        cfg = {}

    fear_thr = cfg.get("vix_fear_threshold", 30)
    greed_thr = cfg.get("vix_greed_threshold", 15)
    window_years = cfg.get("ratio_or_argent_fenetre_percentile_ans", 20)
    seuil_achat = cfg.get("signal_macro_seuil_achat", 3)

    s_taux = score_taux_reels(taux_reel_dernier)
    s_dxy = score_dxy_trend(close_dxy)
    s_vix = score_vix(vix_level, fear_thr, greed_thr)
    s_rsi = score_rsi_or(rsi_or)
    s_ratio = score_ratio_or_argent(ratio_or_argent, ratio_series, window_years)

    total = s_taux + s_dxy + s_vix + s_rsi + s_ratio

    if total >= seuil_achat:
        verdict = "favorable"
    elif total >= 0:
        verdict = "neutre"
    else:
        verdict = "défavorable"

    return {
        "score_total": round(total, 2),
        "verdict": verdict,
        "details": {
            "taux_reels": s_taux,
            "dxy_trend": s_dxy,
            "vix_level": s_vix,
            "rsi_or": s_rsi,
            "ratio_or_argent": s_ratio,
        },
    }

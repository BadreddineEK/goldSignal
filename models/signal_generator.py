"""
signal_generator.py — Générateur de signal actionnable pour GoldSignal.

Prend les probabilités de tendance (issues des modèles ML/ARIMA) et les
combine avec le score macro pour produire un signal discret :
  ✅ ACHAT suggéré | ⏳ ATTENDRE | ❌ VENDRE suggéré

Sortie du signal :
  Probabilité de tendance (haussière / neutre / baissière) à l'horizon cible.
  PAS un prix cible — conformité halal (pas de levier, pas de dérivés).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Labels humains
_LABELS = {1: "Haussier 📈", 0: "Neutre ↔️", -1: "Baissier 📉"}
_COLORS = {1: "#22c55e", 0: "#f59e0b", -1: "#ef4444"}


# ---------------------------------------------------------------------------
# Conversion probabilités → signal discret
# ---------------------------------------------------------------------------

def probabilities_to_signal(p_haussier: float, p_neutre: float,
                              p_baissier: float,
                              conviction_threshold: float = 0.45) -> dict:
    """Convertit les probabilités de classe en signal de tendance.

    Args:
        p_haussier: Probabilité de tendance haussière (0–1).
        p_neutre: Probabilité neutre.
        p_baissier: Probabilité baissière.
        conviction_threshold: Seuil de probabilité pour avoir une conviction
                               (signal non neutre). Défaut : 0.45.

    Returns:
        Dict avec :
          - 'direction': {'haussier': float, 'neutre': float, 'baissier': float}
          - 'signal': 1 | 0 | -1
          - 'label': str
          - 'conviction': float (p_dominant - 1/3) normalisé
          - 'color': couleur hex
    """
    probs = {"haussier": p_haussier, "neutre": p_neutre, "baissier": p_baissier}
    dominant = max(probs, key=probs.get)
    p_dominant = probs[dominant]

    if p_dominant < conviction_threshold:
        signal = 0
    elif dominant == "haussier":
        signal = 1
    elif dominant == "baissier":
        signal = -1
    else:
        signal = 0

    conviction = max(0.0, (p_dominant - 1 / 3) / (2 / 3))  # normalisation [0,1]

    return {
        "direction": probs,
        "signal": signal,
        "label": _LABELS[signal],
        "conviction": round(conviction, 3),
        "color": _COLORS[signal],
    }


# ---------------------------------------------------------------------------
# Signal global : combinaison modèle + macro + prime
# ---------------------------------------------------------------------------

def generate_actionable_signal(
    ml_signal: dict,
    macro_verdict: str,
    prime_pct: Optional[float] = None,
    spread_pct: Optional[float] = None,
    prime_seuil_achat: float = 5.0,
    pl_seuil_vente_pct: float = 20.0,
    pl_actuel_pct: Optional[float] = None,
) -> dict:
    """Génère le signal d'action combiné (achat / attendre / vendre).

    Règle (voir cahier des charges §3.5) :
      ACHAT : signal_haussier ET macro_favorable ET prime+spread ≤ seuil
      VENTE : signal_baissier ET P&L latent ≥ seuil
      SINON : ATTENDRE

    Args:
        ml_signal: Résultat de probabilities_to_signal().
        macro_verdict: 'favorable' | 'neutre' | 'défavorable'.
        prime_pct: Prime actuelle en % (optionnel).
        spread_pct: Spread actuel en % (optionnel).
        prime_seuil_achat: Seuil max de prime pour déclencher achat.
        pl_seuil_vente_pct: P&L minimum pour déclencher signal vente.
        pl_actuel_pct: P&L latent actuel du portefeuille en % (optionnel).

    Returns:
        Dict avec :
          - 'action': 'ACHAT' | 'VENDRE' | 'ATTENDRE'
          - 'action_color': couleur hex
          - 'raisons': liste de str expliquant la décision
          - 'ml': signal ML
          - 'macro': verdict macro
    """
    raisons = []
    action = "ATTENDRE"

    signal = ml_signal.get("signal", 0)
    conviction = ml_signal.get("conviction", 0.0)

    # --- Condition ACHAT ---
    cond_ml_haussier = signal == 1 and conviction > 0.2
    cond_macro_ok = macro_verdict == "favorable"
    cond_prime_ok = (prime_pct is not None and spread_pct is not None
                     and (prime_pct + spread_pct) <= prime_seuil_achat)

    if cond_ml_haussier and cond_macro_ok:
        if prime_pct is not None:
            if cond_prime_ok:
                action = "ACHAT"
                raisons.append(f"✅ Signal ML haussier (conviction {conviction:.0%})")
                raisons.append(f"✅ Contexte macro favorable")
                raisons.append(f"✅ Score prime+spread ({prime_pct+spread_pct:.1f}%) ≤ seuil ({prime_seuil_achat:.1f}%)")
            else:
                action = "ATTENDRE"
                raisons.append(f"✅ Signal ML haussier, contexte favorable")
                raisons.append(f"⚠️ Mais prime+spread ({prime_pct+spread_pct:.1f}%) trop élevé (> {prime_seuil_achat:.1f}%)")
        else:
            action = "ACHAT"
            raisons.append(f"✅ Signal ML haussier (conviction {conviction:.0%})")
            raisons.append(f"✅ Contexte macro favorable")
            raisons.append("ℹ️ Vérifiez la prime du comptoir avant d'acheter")

    # --- Condition VENTE ---
    elif signal == -1 and pl_actuel_pct is not None and pl_actuel_pct >= pl_seuil_vente_pct:
        action = "VENDRE"
        raisons.append(f"📉 Signal ML baissier (conviction {conviction:.0%})")
        raisons.append(f"💰 P&L latent {pl_actuel_pct:.1f}% ≥ seuil de vente ({pl_seuil_vente_pct:.1f}%)")

    # --- ATTENDRE explicite ---
    else:
        if signal == 0:
            raisons.append("⏳ Signal ML neutre — pas de conviction suffisante")
        elif signal == -1:
            raisons.append(f"📉 Signal ML baissier — éviter d'acheter")
        elif signal == 1 and not cond_macro_ok:
            raisons.append(f"✅ Signal ML haussier mais contexte macro {macro_verdict}")
        if not cond_prime_ok and prime_pct is not None:
            raisons.append(f"⚠️ Prime+spread ({prime_pct + (spread_pct or 0):.1f}%) au-dessus du seuil")

    action_colors = {"ACHAT": "#22c55e", "VENDRE": "#ef4444", "ATTENDRE": "#f59e0b"}

    return {
        "action": action,
        "action_color": action_colors[action],
        "raisons": raisons,
        "ml": ml_signal,
        "macro": macro_verdict,
    }


# ---------------------------------------------------------------------------
# Prédiction temps réel (dernier signal à partir du modèle entraîné)
# ---------------------------------------------------------------------------

def get_latest_rf_signal(
    fitted: dict,
    X_latest: pd.DataFrame,
) -> dict:
    """Calcule le signal RF sur les dernières features disponibles.

    Args:
        fitted: Modèle RF entraîné (dict de train_random_forest).
        X_latest: DataFrame des dernières features (1 ligne ou plus).

    Returns:
        Dict signal (voir probabilities_to_signal).
    """
    from models.ml_models import predict_rf_probabilities

    proba_df = predict_rf_probabilities(fitted, X_latest.tail(1))

    p_h = float(proba_df.get("p_haussier", 0).iloc[0]) if "p_haussier" in proba_df else 1/3
    p_n = float(proba_df.get("p_neutre", 0).iloc[0]) if "p_neutre" in proba_df else 1/3
    p_b = float(proba_df.get("p_baissier", 0).iloc[0]) if "p_baissier" in proba_df else 1/3

    return probabilities_to_signal(p_h, p_n, p_b)


def build_signal_history(predictions: pd.Series) -> pd.DataFrame:
    """Construit l'historique des signaux walk-forward (out-of-sample).

    Args:
        predictions: Série de prédictions {-1, 0, 1} indexée par date.

    Returns:
        DataFrame avec colonnes ['date', 'signal', 'label', 'color'].
    """
    df = predictions.reset_index()
    df.columns = ["date", "signal"]
    df["signal"] = df["signal"].round().astype(int)
    df["label"] = df["signal"].map(_LABELS)
    df["color"] = df["signal"].map(_COLORS)
    return df

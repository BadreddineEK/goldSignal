"""
model_store.py — Sauvegarde & chargement des résultats ML pré-entraînés.

Permet de persister les métriques, prédictions et signaux entre sessions
et de les déployer sur Streamlit Cloud sans relancer l'entraînement.

Format : pickle compressé (.pkl.gz) dans models/pretrained/
"""
import gzip
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Répertoire de stockage des modèles pré-entraînés
_PRETRAINED_DIR = Path(__file__).parent / "pretrained"


def is_cloud() -> bool:
    """Détecte si l'app tourne sur Streamlit Community Cloud."""
    return (
        os.path.exists("/mount/src")                              # Streamlit Cloud mount point
        or os.environ.get("HOME", "").startswith("/home/appuser")  # user Streamlit Cloud
        or os.environ.get("STREAMLIT_CLOUD", "") == "1"           # variable explicite
    )


def _store_path(horizon: int) -> Path:
    """Retourne le chemin du fichier pkl.gz pour un horizon donné."""
    _PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    return _PRETRAINED_DIR / f"ml_results_h{horizon}.pkl.gz"


def save_pretrained(
    horizon: int,
    ml_results: dict,
    rw_metrics: dict,
    fi_df: pd.DataFrame,
    latest_signal: Optional[dict],
    lstm_result: Optional[dict],
    hybrid_result: Optional[dict],
    seuil_direction: float,
    n_splits: int,
    timestamp: str = "",
) -> bool:
    """Sauvegarde les résultats d'entraînement.

    Args:
        horizon:          Horizon de prédiction utilisé.
        ml_results:       Dict résultats RF/XGB.
        rw_metrics:       Métriques Random Walk baseline.
        fi_df:            DataFrame feature importance.
        latest_signal:    Signal temps réel RF.
        lstm_result:      Résultats LSTM (ou None).
        hybrid_result:    Résultats hybride (ou None).
        seuil_direction:  Seuil ternaire utilisé.
        n_splits:         Nombre de folds.
        timestamp:        Date/heure d'entraînement.

    Returns:
        True si sauvegarde réussie.
    """
    payload = {
        "horizon":           horizon,
        "seuil_direction":   seuil_direction,
        "n_splits":          n_splits,
        "ml_results":        ml_results,
        "rw_metrics":        rw_metrics,
        "fi_df":             fi_df,
        "latest_signal":     latest_signal,
        "lstm_result":       lstm_result,
        "hybrid_result":     hybrid_result,
        "timestamp":         timestamp,
    }
    try:
        path = _store_path(horizon)
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Modèle pré-entraîné sauvegardé : %s (%.1f KB)", path, path.stat().st_size / 1024)
        return True
    except Exception as exc:
        logger.warning("Impossible de sauvegarder le modèle : %s", exc)
        return False


def load_pretrained(horizon: int) -> Optional[dict]:
    """Charge les résultats pré-entraînés pour un horizon.

    Args:
        horizon: Horizon de prédiction.

    Returns:
        Dict payload ou None si introuvable.
    """
    path = _store_path(horizon)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info("Modèle pré-entraîné chargé : %s (entr. le %s)", path, payload.get("timestamp", "?"))
        return payload
    except Exception as exc:
        logger.warning("Impossible de charger le modèle pré-entraîné : %s", exc)
        return None


def list_pretrained() -> list[dict]:
    """Liste tous les modèles pré-entraînés disponibles.

    Returns:
        Liste de dicts {horizon, timestamp, path, size_kb}.
    """
    if not _PRETRAINED_DIR.exists():
        return []
    results = []
    for f in sorted(_PRETRAINED_DIR.glob("ml_results_h*.pkl.gz")):
        try:
            horizon = int(f.stem.split("_h")[1].split(".")[0])
            payload = load_pretrained(horizon)
            results.append({
                "horizon":   horizon,
                "timestamp": payload.get("timestamp", "?") if payload else "?",
                "path":      str(f),
                "size_kb":   round(f.stat().st_size / 1024, 1),
            })
        except Exception:
            continue
    return results


def has_pretrained(horizon: int) -> bool:
    """Vérifie si un modèle pré-entraîné existe pour cet horizon."""
    return _store_path(horizon).exists()

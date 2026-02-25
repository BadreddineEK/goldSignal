"""
export.py — Export des données GoldSignal en JSON et CSV.

Permet d'exporter / importer la configuration complète et l'historique
des achats (portefeuille).
"""

import json
import csv
import io
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------

def export_config_json(config: dict) -> str:
    """Sérialise la configuration complète en JSON.

    Args:
        config: Dictionnaire de configuration (issu de load_full_config_from_db).

    Returns:
        Chaîne JSON indentée.
    """
    return json.dumps(config, ensure_ascii=False, indent=2)


def import_config_json(json_str: str) -> dict:
    """Désérialise une configuration depuis un JSON.

    Args:
        json_str: Chaîne JSON.

    Returns:
        Dictionnaire de configuration.

    Raises:
        ValueError: Si le JSON est invalide.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide : {exc}") from exc


# ---------------------------------------------------------------------------
# Export CSV portefeuille
# ---------------------------------------------------------------------------

def export_portfolio_csv(portfolio: list[dict]) -> str:
    """Exporte le portefeuille en CSV (compatible Excel FR).

    Args:
        portfolio: Liste des achats (format get_portfolio()).

    Returns:
        Chaîne CSV avec séparateur ';' pour compatibilité Excel.
    """
    if not portfolio:
        return ""

    fieldnames = [
        "id", "date_achat", "piece_nom", "metal",
        "quantite", "prix_ask", "g_fin",
        "comptoir", "note", "created_at",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        delimiter=";",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in portfolio:
        writer.writerow(row)

    return output.getvalue()


def export_portfolio_json(portfolio: list[dict]) -> str:
    """Exporte le portefeuille en JSON.

    Args:
        portfolio: Liste des achats.

    Returns:
        Chaîne JSON indentée.
    """
    return json.dumps(portfolio, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Nommage des fichiers d'export
# ---------------------------------------------------------------------------

def export_filename(prefix: str, ext: str = "json") -> str:
    """Génère un nom de fichier d'export daté.

    Args:
        prefix: Préfixe du fichier (ex: 'goldsignal_config').
        ext: Extension (défaut 'json').

    Returns:
        Nom de fichier ex: 'goldsignal_config_2026-02-25.json'.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{prefix}_{today}.{ext}"

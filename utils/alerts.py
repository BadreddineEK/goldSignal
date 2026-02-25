"""
alerts.py — Logique des alertes configurables pour GoldSignal.

Évalue les conditions d'alerte à partir des données marché et du portefeuille,
et enregistre les alertes dans la base SQLite si les seuils sont dépassés.
"""

import logging
from datetime import date
from typing import Optional

from data.database import add_alert, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Évaluation des conditions
# ---------------------------------------------------------------------------

def check_prime_alerte(piece_nom: str, prime_pct: float) -> None:
    """Déclenche une alerte achat si la prime est sous le seuil configuré.

    Args:
        piece_nom: Nom de la pièce.
        prime_pct: Prime actuelle en %.
    """
    cfg = get_config("portfolio") or {}
    seuil_prime = cfg.get("alerte_prime_achat_max_pct", 3.0)

    if prime_pct <= seuil_prime:
        msg = (
            f"ACHAT potentiel — {piece_nom} : prime {prime_pct:.2f}% "
            f"≤ seuil {seuil_prime:.1f}%"
        )
        add_alert("achat", msg)
        logger.info("Alerte achat créée : %s", msg)


def check_pl_alerte(piece_nom: str, pl_pct: float,
                    quantite: float, valeur_eur: float) -> None:
    """Déclenche une alerte vente si le P&L dépasse le seuil configuré.

    Args:
        piece_nom: Nom de la pièce.
        pl_pct: Plus-value latente en %.
        quantite: Nombre de pièces concernées.
        valeur_eur: Valeur de la position en €.
    """
    cfg = get_config("portfolio") or {}
    seuil_pl = cfg.get("alerte_pl_vente_pct", 20.0)

    if pl_pct >= seuil_pl:
        msg = (
            f"VENTE suggérée — {piece_nom} : P&L {pl_pct:.1f}% "
            f"({quantite:.0f} pièces, valeur {valeur_eur:.0f} €)"
        )
        add_alert("vente", msg)
        logger.info("Alerte vente créée : %s", msg)


def check_macro_alerte(score_total: float, verdict: str) -> None:
    """Déclenche une alerte info quand le contexte macro change de catégorie.

    Args:
        score_total: Score macro composite (-5 à +5).
        verdict: 'favorable' | 'neutre' | 'défavorable'.
    """
    if verdict == "favorable":
        msg = f"Contexte macro FAVORABLE à l'achat (score {score_total:+.1f}/5)"
        add_alert("info", msg)


def evaluate_all_alerts(portfolio: list[dict],
                        spots: dict,
                        macro_score: Optional[dict] = None) -> None:
    """Évalue toutes les alertes portefeuille + macro en une passe.

    À appeler quotidiennement au chargement des pages.

    Args:
        portfolio: Liste des achats (format get_portfolio()).
        spots: Dict {'xau_eur_g': float, 'xag_eur_g': float}.
        macro_score: Résultat de compute_macro_score() (optionnel).
    """
    spot_or = spots.get("xau_eur_g")
    spot_ag = spots.get("xag_eur_g")

    for achat in portfolio:
        metal = achat.get("metal")
        spot = spot_or if metal == "or" else spot_ag
        if spot is None:
            continue

        g_fin = achat.get("g_fin", 0)
        quantite = achat.get("quantite", 1)
        prix_ask_achat = achat.get("prix_ask", 0)
        piece_nom = achat.get("piece_nom", "?")

        valeur_spot_piece = spot * g_fin
        valeur_totale = valeur_spot_piece * quantite
        cout_total = prix_ask_achat * quantite
        pl_pct = (valeur_spot_piece / prix_ask_achat - 1) * 100 if prix_ask_achat else 0

        check_pl_alerte(piece_nom, pl_pct, quantite, valeur_totale)

    # Alerte macro
    if macro_score:
        check_macro_alerte(macro_score["score_total"], macro_score["verdict"])

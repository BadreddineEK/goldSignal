"""
formatting.py — Fonctions de formatage pour l'affichage dans GoldSignal.

Fournit des helpers pour formater les prix, pourcentages, couleurs
et verdicts à afficher dans l'interface Streamlit.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Formatage monétaire et pourcentages
# ---------------------------------------------------------------------------

def fmt_eur(value: Optional[float], decimals: int = 2) -> str:
    """Formate un montant en euros.

    Args:
        value: Valeur numérique.
        decimals: Nombre de décimales (défaut 2).

    Returns:
        Chaîne formatée ex: '1 234,56 €' ou '—' si None/NaN.
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "—"
    fmt = f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"{fmt} €"


def fmt_pct(value: Optional[float], decimals: int = 2, sign: bool = True) -> str:
    """Formate un pourcentage.

    Args:
        value: Valeur en % (ex: 2.5 pour 2,5%).
        decimals: Décimales (défaut 2).
        sign: Si True, affiche + pour les valeurs positives.

    Returns:
        Chaîne formatée ex: '+2,50 %' ou '—'.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    prefix = "+" if sign and value > 0 else ""
    formatted = f"{value:.{decimals}f}".replace(".", ",")
    return f"{prefix}{formatted} %"


def fmt_g(value: Optional[float], decimals: int = 3) -> str:
    """Formate un poids en grammes.

    Args:
        value: Valeur en grammes.
        decimals: Décimales (défaut 3).

    Returns:
        Chaîne formatée ex: '5,806 g'.
    """
    if value is None:
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",") + " g"


def fmt_ratio(value: Optional[float], decimals: int = 1) -> str:
    """Formate un ratio (sans unité).

    Args:
        value: Valeur du ratio.
        decimals: Décimales (défaut 1).

    Returns:
        Chaîne formatée ex: '83,4'.
    """
    if value is None:
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Couleurs et verdicts
# ---------------------------------------------------------------------------

def verdict_color(verdict: str) -> str:
    """Retourne la couleur CSS associée à un verdict.

    Args:
        verdict: 'good' | 'warn' | 'bad' | 'favorable' | 'neutre' | 'défavorable'.

    Returns:
        Couleur hex CSS.
    """
    mapping = {
        "good": "#22c55e",       # vert
        "favorable": "#22c55e",
        "warn": "#f59e0b",       # orange
        "neutre": "#f59e0b",
        "bad": "#ef4444",        # rouge
        "défavorable": "#ef4444",
    }
    return mapping.get(verdict.lower(), "#6b7280")  # gris par défaut


def verdict_emoji(verdict: str) -> str:
    """Retourne l'emoji associé à un verdict.

    Args:
        verdict: 'good' | 'warn' | 'bad' | 'favorable' | 'neutre' | 'défavorable'.

    Returns:
        Emoji str.
    """
    mapping = {
        "good": "✅",
        "favorable": "✅",
        "warn": "⚠️",
        "neutre": "⚠️",
        "bad": "❌",
        "défavorable": "❌",
    }
    return mapping.get(verdict.lower(), "❓")


def verdict_label_fr(verdict: str) -> str:
    """Retourne le libellé français d'un verdict.

    Args:
        verdict: Clé de verdict.

    Returns:
        Libellé en français.
    """
    mapping = {
        "good": "Bon prix",
        "warn": "Prix correct",
        "bad": "Trop cher",
        "favorable": "Favorable",
        "neutre": "Neutre",
        "défavorable": "Défavorable",
    }
    return mapping.get(verdict.lower(), verdict)


def colored_metric(label: str, value: str, verdict: str) -> str:
    """Génère un bloc HTML coloré pour une métrique Streamlit.

    À utiliser avec st.markdown(..., unsafe_allow_html=True).

    Args:
        label: Libellé de la métrique.
        value: Valeur formatée.
        verdict: Verdict ('good' | 'warn' | 'bad').

    Returns:
        HTML string.
    """
    color = verdict_color(verdict)
    emoji = verdict_emoji(verdict)
    return (
        f'<div style="background:{color}20; border-left:4px solid {color}; '
        f'padding:8px 12px; border-radius:4px; margin:4px 0">'
        f'<small style="color:{color}; font-weight:600">{label}</small><br>'
        f'<span style="font-size:1.3em; font-weight:700; color:{color}">'
        f'{emoji} {value}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Calculs prime / spread / score (formules métier)
# ---------------------------------------------------------------------------

def compute_prime(ask: float, g_fin: float, spot_eur_g: float) -> Optional[float]:
    """Calcule la prime % = (Ask/g_fin) / spot_eur_g - 1.

    Args:
        ask: Prix Ask en € de la pièce.
        g_fin: Poids en grammes fins.
        spot_eur_g: Cours spot en €/g fin.

    Returns:
        Prime en %, ou None si données invalides.
    """
    if not g_fin or not spot_eur_g or spot_eur_g == 0:
        return None
    return (ask / g_fin / spot_eur_g - 1) * 100


def compute_spread(ask: float, bid: float) -> Optional[float]:
    """Calcule le spread % = (Ask - Bid) / Ask.

    Args:
        ask: Prix Ask.
        bid: Prix Bid.

    Returns:
        Spread en %, ou None si données invalides.
    """
    if not ask or ask == 0:
        return None
    return ((ask - bid) / ask) * 100


def compute_score(prime_pct: Optional[float],
                  spread_pct: Optional[float]) -> Optional[float]:
    """Calcule le score = Prime% + Spread%.

    Args:
        prime_pct: Prime en %.
        spread_pct: Spread en %.

    Returns:
        Score total, ou None si données manquantes.
    """
    if prime_pct is None or spread_pct is None:
        return None
    return prime_pct + spread_pct


def get_verdict(value: float, good_max: float,
                warn_max: float) -> str:
    """Retourne le verdict selon les seuils configurés.

    Args:
        value: Valeur à évaluer (prime%, spread% ou score).
        good_max: Seuil maximum pour 'good'.
        warn_max: Seuil maximum pour 'warn'.

    Returns:
        'good' | 'warn' | 'bad'.
    """
    if value <= good_max:
        return "good"
    elif value <= warn_max:
        return "warn"
    else:
        return "bad"

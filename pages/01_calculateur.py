"""
01_calculateur.py — Calculateur Terrain GoldSignal (Module 1).

Fonctionnalités :
  - Saisie Ask/Bid pour chaque pièce du catalogue
  - Récupération automatique du spot or/argent (yfinance) ou saisie manuelle
  - Calcul : Prime%, Spread%, Score = Prime + Spread
  - Verdict coloré configurable (seuils par pièce/métal)
  - Règle relative : surcoût 10F vs 20F
  - Export JSON de la session
"""

import streamlit as st
import pandas as pd
from typing import Optional

from data.database import get_pieces, get_seuils_piece, get_config
from data.fetcher import get_spot_xau_eur, get_spot_xag_eur
from utils.formatting import (
    fmt_eur, fmt_pct, compute_prime, compute_spread, compute_score,
    get_verdict, verdict_emoji, verdict_label_fr, colored_metric,
)
from utils.export import export_config_json, export_filename

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.header("🧮 Calculateur Terrain")
st.caption("Évaluez instantanément si un prix comptoir est bon, correct ou cher.")

# ---------------------------------------------------------------------------
# Récupération des spots
# ---------------------------------------------------------------------------
with st.expander("📡 Spots de référence", expanded=True):
    col_mode, col_refresh = st.columns([3, 1])
    with col_mode:
        mode_spot = st.radio(
            "Source du spot",
            ["🔄 Automatique (yfinance)", "✏️ Manuel"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_refresh:
        refresh_btn = st.button("🔄 Actualiser", use_container_width=True)

    col1, col2 = st.columns(2)

    if "🔄 Automatique" in mode_spot:
        if refresh_btn or "spot_xau_g" not in st.session_state:
            with st.spinner("Récupération des spots…"):
                st.session_state["spot_xau_g"] = get_spot_xau_eur(force_refresh=refresh_btn)
                st.session_state["spot_xag_g"] = get_spot_xag_eur(force_refresh=refresh_btn)

        spot_xau_g: Optional[float] = st.session_state.get("spot_xau_g")
        spot_xag_g: Optional[float] = st.session_state.get("spot_xag_g")

        with col1:
            if spot_xau_g:
                st.metric("🥇 Or", f"{fmt_eur(spot_xau_g)} /g fin",
                          help="Cours GC=F converti en €/g fin via EURUSD=X")
            else:
                st.warning("Spot or indisponible")
        with col2:
            if spot_xag_g:
                st.metric("🥈 Argent", f"{fmt_eur(spot_xag_g, 4)} /g fin",
                          help="Cours SI=F converti en €/g fin via EURUSD=X")
            else:
                st.warning("Spot argent indisponible")
    else:
        with col1:
            spot_xau_g = st.number_input(
                "🥇 Spot Or (€/g fin)",
                min_value=0.0, value=60.0, step=0.1, format="%.4f",
            )
        with col2:
            spot_xag_g = st.number_input(
                "🥈 Spot Argent (€/g fin)",
                min_value=0.0, value=0.85, step=0.01, format="%.4f",
            )

st.markdown("---")

# ---------------------------------------------------------------------------
# Chargement des pièces
# ---------------------------------------------------------------------------
pieces = get_pieces(actif_only=True)
if not pieces:
    st.error("Aucune pièce dans le catalogue. Vérifiez la page Config.")
    st.stop()

pieces_or = [p for p in pieces if p["metal"] == "or"]
pieces_ag = [p for p in pieces if p["metal"] == "argent"]

# ---------------------------------------------------------------------------
# Résultats de la session (accumulés pour export)
# ---------------------------------------------------------------------------
session_results = []

# ---------------------------------------------------------------------------
# Helper : affichage d'une pièce
# ---------------------------------------------------------------------------

def afficher_piece(piece: dict, spot_g: Optional[float]) -> Optional[dict]:
    """Affiche le bloc de calcul pour une pièce et retourne les résultats."""
    piece_id = piece["id"]
    piece_nom = piece["nom"]
    g_fin = piece["g_fin"]

    seuils = get_seuils_piece(piece_id)
    cfg_metal = get_config("seuils") or {}
    metal_cfg = cfg_metal.get(piece["metal"], {})
    prime_good = (seuils or {}).get("prime_good_max", metal_cfg.get("prime_good_max", 2.0))
    prime_warn = (seuils or {}).get("prime_warn_max", metal_cfg.get("prime_warn_max", 5.0))
    spread_good = (seuils or {}).get("spread_good_max", metal_cfg.get("spread_good_max", 2.0))
    spread_warn = (seuils or {}).get("spread_warn_max", metal_cfg.get("spread_warn_max", 4.0))
    score_good = (seuils or {}).get("score_good_max", metal_cfg.get("score_good_max", 4.0))
    score_warn = (seuils or {}).get("score_warn_max", metal_cfg.get("score_warn_max", 8.0))

    with st.container(border=True):
        st.markdown(f"**{piece_nom}** — {g_fin:.3f} g fin")

        c1, c2, c3 = st.columns(3)
        spot_piece = (spot_g * g_fin) if spot_g else 0.0
        with c1:
            ask = st.number_input(
                "Prix Ask (€)",
                key=f"ask_{piece_id}",
                min_value=0.0,
                value=round(spot_piece * 1.03, 2) if spot_piece else 0.0,
                step=0.5, format="%.2f",
            )
        with c2:
            bid = st.number_input(
                "Prix Bid (€)",
                key=f"bid_{piece_id}",
                min_value=0.0,
                value=round(spot_piece * 0.97, 2) if spot_piece else 0.0,
                step=0.5, format="%.2f",
            )
        with c3:
            if spot_g:
                st.metric("Spot pièce ≈", fmt_eur(spot_piece))
            else:
                st.caption("Spot non disponible")

        if ask > 0 and bid > 0 and spot_g:
            prime = compute_prime(ask, g_fin, spot_g)
            spread = compute_spread(ask, bid)
            score = compute_score(prime, spread)

            v_prime = get_verdict(prime, prime_good, prime_warn)
            v_spread = get_verdict(spread, spread_good, spread_warn)
            v_score = get_verdict(score, score_good, score_warn)

            res_cols = st.columns(3)
            with res_cols[0]:
                st.markdown(
                    colored_metric("Prime", fmt_pct(prime), v_prime),
                    unsafe_allow_html=True,
                )
            with res_cols[1]:
                st.markdown(
                    colored_metric("Spread", fmt_pct(spread), v_spread),
                    unsafe_allow_html=True,
                )
            with res_cols[2]:
                st.markdown(
                    colored_metric(
                        "Score total",
                        f"{verdict_emoji(v_score)} {fmt_pct(score)}",
                        v_score,
                    ),
                    unsafe_allow_html=True,
                )

            return {
                "piece": piece_nom,
                "metal": piece["metal"],
                "g_fin": g_fin,
                "ask": ask,
                "bid": bid,
                "spot_g": spot_g,
                "spot_piece": spot_piece,
                "prime_pct": round(prime, 3),
                "spread_pct": round(spread, 3),
                "score": round(score, 3),
                "verdict": v_score,
            }
        elif ask > 0 and bid == 0:
            st.caption("Saisissez le Bid pour calculer le spread.")
        elif ask == 0:
            st.caption("Saisissez le prix Ask pour démarrer le calcul.")
        return None


# ---------------------------------------------------------------------------
# Section OR
# ---------------------------------------------------------------------------
if pieces_or:
    st.subheader("🥇 Pièces en or")
    for piece in pieces_or:
        res = afficher_piece(piece, spot_xau_g)
        if res:
            session_results.append(res)

    # Règle relative : surcoût Napoléon 10F vs 20F
    nap10_res = next((r for r in session_results if "10F" in r["piece"]), None)
    nap20_res = next((r for r in session_results if "20F" in r["piece"]), None)
    if nap10_res and nap20_res:
        surcout = nap10_res["prime_pct"] - nap20_res["prime_pct"]
        emoji = "⚠️" if abs(surcout) > 2 else "✅"
        st.info(
            f"{emoji} Surcoût Napoléon 10F vs 20F : **{fmt_pct(surcout)}** "
            f"(positif = 10F plus cher en prime relative)"
        )

# ---------------------------------------------------------------------------
# Section ARGENT
# ---------------------------------------------------------------------------
if pieces_ag:
    st.subheader("🥈 Pièces en argent")
    for piece in pieces_ag:
        res = afficher_piece(piece, spot_xag_g)
        if res:
            session_results.append(res)

# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------
if session_results:
    st.markdown("---")
    st.subheader("📋 Récapitulatif de session")

    df = pd.DataFrame(session_results)[
        ["piece", "ask", "bid", "spot_piece", "prime_pct", "spread_pct", "score", "verdict"]
    ]
    df.columns = ["Pièce", "Ask (€)", "Bid (€)", "Spot pièce (€)", "Prime %", "Spread %", "Score", "Verdict"]
    df["Verdict"] = df["Verdict"].apply(
        lambda v: f"{verdict_emoji(v)} {verdict_label_fr(v)}"
    )

    st.dataframe(
        df.style.format({
            "Ask (€)": "{:.2f}",
            "Bid (€)": "{:.2f}",
            "Spot pièce (€)": "{:.2f}",
            "Prime %": "{:.2f}",
            "Spread %": "{:.2f}",
            "Score": "{:.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    # Export JSON
    json_export = export_config_json({"session": session_results})
    st.download_button(
        "⬇️ Exporter la session (JSON)",
        data=json_export,
        file_name=export_filename("goldsignal_session"),
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# Aide / Formules
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Formules utilisées"):
    st.markdown("""
    | Formule | Définition |
    |---|---|
    | **Prime %** | `(Ask / g_fin) / Spot_€/g − 1` |
    | **Spread %** | `(Ask − Bid) / Ask` |
    | **Score** | `Prime% + Spread%` |

    Le spot est en **€/g fin**. La prime compare le prix Ask ramené au gramme fin
    par rapport au cours spot. Le spread reflète la différence achat/vente du revendeur.

    **Verdicts par défaut (or) :**
    - ✅ Bon : Score ≤ 4 %
    - ⚠️ Correct : Score ≤ 8 %
    - ❌ Trop cher : Score > 8 %

    > Modifiables dans la page **Config ⚙️**.
    """)

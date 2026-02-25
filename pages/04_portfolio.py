"""
04_portfolio.py — Portefeuille Personnel GoldSignal (Module 4).

Fonctionnalités :
  - Saisie d'un achat (date, pièce, quantité, prix Ask, comptoir)
  - Tableau des achats avec P&L temps réel
  - Graphe d'évolution valeur portefeuille vs spot
  - Allocation or/argent actuelle vs cible
  - Alertes P&L configurables
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from data.database import (
    get_pieces, get_portfolio, add_achat, delete_achat, get_config,
)
from data.fetcher import get_spot_xau_eur, get_spot_xag_eur
from utils.formatting import fmt_eur, fmt_pct, verdict_emoji
from utils.alerts import check_pl_alerte
from utils.export import export_portfolio_csv, export_portfolio_json, export_filename

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("💼 Portefeuille")
st.caption("Suivi de vos achats de métaux précieux et P&L temps réel.")

# ---------------------------------------------------------------------------
# Spots actuels
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_spots():
    return {
        "xau_eur_g": get_spot_xau_eur(),
        "xag_eur_g": get_spot_xag_eur(),
    }

spots = get_spots()
spot_or = spots.get("xau_eur_g")
spot_ag = spots.get("xag_eur_g")

# ---------------------------------------------------------------------------
# Formulaire ajout d'achat
# ---------------------------------------------------------------------------
with st.expander("➕ Ajouter un achat", expanded=False):
    pieces = get_pieces(actif_only=True)
    piece_options = {p["nom"]: p for p in pieces}

    with st.form("form_achat", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            piece_nom_sel = st.selectbox("Pièce", list(piece_options.keys()))
            quantite = st.number_input("Quantité", min_value=0.1, value=1.0, step=0.5)
        with col2:
            prix_ask = st.number_input("Prix Ask payé (€)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
            date_achat = st.date_input("Date d'achat", value=date.today())
        with col3:
            comptoir = st.text_input("Comptoir (optionnel)")
            note = st.text_input("Note (optionnel)")

        submitted = st.form_submit_button("💾 Enregistrer l'achat", use_container_width=True)
        if submitted:
            if prix_ask <= 0:
                st.error("Le prix Ask doit être > 0.")
            else:
                piece = piece_options[piece_nom_sel]
                add_achat(
                    date_achat=date_achat.isoformat(),
                    piece_id=piece["id"],
                    quantite=quantite,
                    prix_ask=prix_ask,
                    comptoir=comptoir,
                    note=note,
                )
                st.success(f"✅ Achat enregistré : {quantite}× {piece_nom_sel} à {fmt_eur(prix_ask)}")
                st.rerun()

# ---------------------------------------------------------------------------
# Chargement du portefeuille
# ---------------------------------------------------------------------------
portfolio = get_portfolio()

if not portfolio:
    st.info("Aucun achat enregistré. Utilisez le formulaire ci-dessus pour ajouter votre premier achat.")
    st.stop()

# ---------------------------------------------------------------------------
# Calcul P&L
# ---------------------------------------------------------------------------
rows = []
for achat in portfolio:
    metal = achat["metal"]
    spot_g = spot_or if metal == "or" else spot_ag
    g_fin = achat["g_fin"]
    quantite = achat["quantite"]
    prix_ask = achat["prix_ask"]

    valeur_spot_piece = (spot_g * g_fin) if spot_g else None
    valeur_actuelle = (valeur_spot_piece * quantite) if valeur_spot_piece else None
    cout_total = prix_ask * quantite
    pl_eur = (valeur_actuelle - cout_total) if valeur_actuelle else None
    pl_pct = ((valeur_spot_piece / prix_ask - 1) * 100) if (valeur_spot_piece and prix_ask) else None

    # Check alerte
    if pl_pct is not None and valeur_actuelle is not None:
        check_pl_alerte(achat["piece_nom"], pl_pct, quantite, valeur_actuelle)

    rows.append({
        "ID": achat["id"],
        "Date": achat["date_achat"],
        "Pièce": achat["piece_nom"],
        "Métal": metal,
        "Qté": quantite,
        "Prix Ask (€)": prix_ask,
        "Coût total (€)": round(cout_total, 2),
        "Spot pièce (€)": round(valeur_spot_piece, 2) if valeur_spot_piece else None,
        "Valeur actuelle (€)": round(valeur_actuelle, 2) if valeur_actuelle else None,
        "P&L (€)": round(pl_eur, 2) if pl_eur is not None else None,
        "P&L (%)": round(pl_pct, 2) if pl_pct is not None else None,
        "Comptoir": achat.get("comptoir", ""),
    })

df_portfolio = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# KPIs globaux
# ---------------------------------------------------------------------------
cout_tot = df_portfolio["Coût total (€)"].sum()
val_tot = df_portfolio["Valeur actuelle (€)"].sum() if df_portfolio["Valeur actuelle (€)"].notna().any() else None
pl_tot_eur = (val_tot - cout_tot) if val_tot else None
pl_tot_pct = ((val_tot / cout_tot - 1) * 100) if (val_tot and cout_tot) else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Coût total", fmt_eur(cout_tot))
k2.metric("Valeur actuelle", fmt_eur(val_tot) if val_tot else "—")
k3.metric("P&L (€)", fmt_eur(pl_tot_eur) if pl_tot_eur is not None else "—",
          delta=f"{pl_tot_eur:+.0f} €" if pl_tot_eur else None)
k4.metric("P&L (%)", fmt_pct(pl_tot_pct) if pl_tot_pct is not None else "—",
          delta=fmt_pct(pl_tot_pct) if pl_tot_pct is not None else None)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tableau détaillé
# ---------------------------------------------------------------------------
st.subheader("📋 Détail des achats")

display_cols = [
    "Date", "Pièce", "Métal", "Qté", "Prix Ask (€)", "Coût total (€)",
    "Spot pièce (€)", "Valeur actuelle (€)", "P&L (€)", "P&L (%)",
]
df_display = df_portfolio[display_cols].copy()

# Coloration P&L
def style_pl(val):
    if pd.isna(val):
        return ""
    color = "#22c55e" if val > 0 else "#ef4444" if val < 0 else ""
    return f"color: {color}; font-weight: 600"

st.dataframe(
    df_display.style
        .applymap(style_pl, subset=["P&L (€)", "P&L (%)"])
        .format({
            "Prix Ask (€)": "{:.2f}",
            "Coût total (€)": "{:.2f}",
            "Spot pièce (€)": "{:.2f}",
            "Valeur actuelle (€)": "{:.2f}",
            "P&L (€)": "{:+.2f}",
            "P&L (%)": "{:+.2f}",
        }, na_rep="—"),
    width="stretch",
    hide_index=True,
)

# Suppression d'un achat
with st.expander("🗑️ Supprimer un achat"):
    del_id = st.number_input("ID de l'achat à supprimer", min_value=1, step=1)
    if st.button("Supprimer", type="secondary"):
        delete_achat(int(del_id))
        st.success(f"Achat #{del_id} supprimé.")
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Allocation or/argent
# ---------------------------------------------------------------------------
st.subheader("🥇 Allocation Or / Argent")
cfg_portfolio = get_config("portfolio") or {}
cible_or = cfg_portfolio.get("cible_allocation_or_pct", 70)
cible_ag = 100 - cible_or

val_or = df_portfolio[df_portfolio["Métal"] == "or"]["Valeur actuelle (€)"].sum()
val_ag = df_portfolio[df_portfolio["Métal"] == "argent"]["Valeur actuelle (€)"].sum()
val_total_alloc = val_or + val_ag

if val_total_alloc > 0:
    pct_or = val_or / val_total_alloc * 100
    pct_ag = val_ag / val_total_alloc * 100

    fig_alloc = go.Figure(go.Pie(
        labels=["🥇 Or", "🥈 Argent"],
        values=[val_or, val_ag],
        hole=0.5,
        marker_colors=["#f59e0b", "#94a3b8"],
    ))
    fig_alloc.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
    )
    col_pie, col_alloc = st.columns([1, 1])
    with col_pie:
        st.plotly_chart(fig_alloc, width="stretch")
    with col_alloc:
        st.metric("Or actuel", fmt_pct(pct_or, 1),
                  delta=f"{pct_or - cible_or:+.1f}% vs cible {cible_or}%")
        st.metric("Argent actuel", fmt_pct(pct_ag, 1),
                  delta=f"{pct_ag - cible_ag:+.1f}% vs cible {cible_ag}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.subheader("📤 Export")
col_e1, col_e2 = st.columns(2)
with col_e1:
    csv_data = export_portfolio_csv(portfolio)
    st.download_button(
        "⬇️ Exporter en CSV (Excel)",
        data=csv_data,
        file_name=export_filename("goldsignal_portfolio", "csv"),
        mime="text/csv",
    )
with col_e2:
    json_data = export_portfolio_json(portfolio)
    st.download_button(
        "⬇️ Exporter en JSON",
        data=json_data,
        file_name=export_filename("goldsignal_portfolio"),
        mime="application/json",
    )

"""
05_config.py — Configuration GoldSignal (Module 5).

Sections :
  - Catalogue des pièces (ajout / modification / désactivation)
  - Seuils Prime/Spread/Score par pièce
  - Paramètres macro (fenêtres, seuils)
  - Clé API FRED
  - Paramètres portefeuille (alertes, allocations cibles)
  - Import / Export JSON de la configuration complète
"""

import json
import streamlit as st

from data.database import (
    get_pieces, upsert_piece, get_seuils_piece, upsert_seuils_piece,
    get_config, set_config, load_full_config_from_db,
)
from utils.export import export_config_json, import_config_json, export_filename

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("⚙️ Configuration")
st.caption("Gérez les pièces, les seuils, les paramètres et la clé API FRED.")

tabs = st.tabs(["🪙 Pièces & Seuils", "📊 Macro", "💼 Portefeuille", "🔑 API", "📤 Import/Export"])

# ---------------------------------------------------------------------------
# Tab 1 : Pièces & Seuils
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Catalogue des pièces")
    pieces = get_pieces(actif_only=False)

    for piece in pieces:
        with st.expander(f"{'✅' if piece['actif'] else '❌'} {piece['nom']} ({piece['metal'].upper()})"):
            with st.form(f"form_piece_{piece['id']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    nom = st.text_input("Nom", value=piece["nom"])
                    metal = st.selectbox("Métal", ["or", "argent"],
                                         index=0 if piece["metal"] == "or" else 1)
                with c2:
                    g_fin = st.number_input("g fin", value=float(piece["g_fin"]),
                                             min_value=0.001, step=0.001, format="%.3f")
                    actif = st.checkbox("Actif", value=bool(piece["actif"]))
                with c3:
                    compare_to = st.text_input("Comparer à (id)", value=piece.get("compare_to") or "")

                st.markdown("**Seuils prime/spread**")
                seuils = get_seuils_piece(piece["id"]) or {}
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    pg = st.number_input("Prime Good max %", value=float(seuils.get("prime_good_max", 2.0)), step=0.5)
                    pw = st.number_input("Prime Warn max %", value=float(seuils.get("prime_warn_max", 5.0)), step=0.5)
                with sc2:
                    sg_v = st.number_input("Spread Good max %", value=float(seuils.get("spread_good_max", 2.0)), step=0.5)
                    sw_v = st.number_input("Spread Warn max %", value=float(seuils.get("spread_warn_max", 4.0)), step=0.5)
                with sc3:
                    sco_g = st.number_input("Score Good max %", value=float(seuils.get("score_good_max", 4.0)), step=0.5)
                    sco_w = st.number_input("Score Warn max %", value=float(seuils.get("score_warn_max", 8.0)), step=0.5)

                if st.form_submit_button("💾 Sauvegarder", use_container_width=True):
                    upsert_piece({
                        "id": piece["id"], "nom": nom, "metal": metal,
                        "g_fin": g_fin, "compare_to": compare_to or None,
                        "actif": int(actif),
                    })
                    upsert_seuils_piece({
                        "piece_id": piece["id"],
                        "prime_good_max": pg, "prime_warn_max": pw,
                        "spread_good_max": sg_v, "spread_warn_max": sw_v,
                        "score_good_max": sco_g, "score_warn_max": sco_w,
                    })
                    st.success(f"✅ {nom} sauvegardé.")

    st.markdown("---")
    st.subheader("➕ Nouvelle pièce")
    with st.form("form_nouvelle_piece", clear_on_submit=True):
        nc1, nc2 = st.columns(2)
        with nc1:
            new_id = st.text_input("ID unique (ex: 'oz_or_maple')")
            new_nom = st.text_input("Nom")
            new_metal = st.selectbox("Métal", ["or", "argent"])
        with nc2:
            new_g_fin = st.number_input("g fin", min_value=0.001, value=31.103, step=0.001, format="%.3f")
            new_compare_to = st.text_input("Comparer à (id, optionnel)")

        if st.form_submit_button("➕ Ajouter la pièce"):
            if not new_id or not new_nom:
                st.error("ID et nom sont requis.")
            else:
                upsert_piece({
                    "id": new_id, "nom": new_nom, "metal": new_metal,
                    "g_fin": new_g_fin, "compare_to": new_compare_to or None,
                    "actif": 1,
                })
                st.success(f"Pièce '{new_nom}' ajoutée.")
                st.rerun()

# ---------------------------------------------------------------------------
# Tab 2 : Paramètres Macro
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Paramètres d'analyse macro")
    cfg_macro = get_config("macro") or {}

    with st.form("form_macro"):
        col1, col2 = st.columns(2)
        with col1:
            rolling_w = st.number_input("Fenêtre corrélation rolling (j)", value=int(cfg_macro.get("rolling_corr_window", 60)), step=5)
            rsi_p = st.number_input("Période RSI", value=int(cfg_macro.get("rsi_period", 14)), step=1)
            bb_w = st.number_input("Fenêtre Bollinger", value=int(cfg_macro.get("bollinger_window", 20)), step=5)
            bb_sigma = st.number_input("Sigma Bollinger", value=float(cfg_macro.get("bollinger_sigma", 2.0)), step=0.5)
        with col2:
            vix_fear = st.number_input("VIX seuil peur", value=float(cfg_macro.get("vix_fear_threshold", 30)), step=1.0)
            vix_greed = st.number_input("VIX seuil cupidité", value=float(cfg_macro.get("vix_greed_threshold", 15)), step=1.0)
            ratio_years = st.number_input("Fenêtre percentile ratio (ans)", value=int(cfg_macro.get("ratio_or_argent_fenetre_percentile_ans", 20)), step=5)
            score_achat = st.number_input("Score macro seuil achat", value=float(cfg_macro.get("signal_macro_seuil_achat", 3)), step=0.5)

        if st.form_submit_button("💾 Sauvegarder paramètres macro"):
            updated = {**cfg_macro,
                       "rolling_corr_window": rolling_w, "rsi_period": rsi_p,
                       "bollinger_window": bb_w, "bollinger_sigma": bb_sigma,
                       "vix_fear_threshold": vix_fear, "vix_greed_threshold": vix_greed,
                       "ratio_or_argent_fenetre_percentile_ans": ratio_years,
                       "signal_macro_seuil_achat": score_achat}
            set_config("macro", updated)
            st.success("✅ Paramètres macro sauvegardés.")
            st.cache_data.clear()

# ---------------------------------------------------------------------------
# Tab 3 : Portefeuille
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Paramètres portefeuille & alertes")
    cfg_p = get_config("portfolio") or {}

    with st.form("form_portfolio"):
        col1, col2 = st.columns(2)
        with col1:
            pl_alerte = st.number_input("Alerte vente si P&L > (%)", value=float(cfg_p.get("alerte_pl_vente_pct", 20.0)), step=1.0)
            prime_alerte = st.number_input("Alerte achat si Prime < (%)", value=float(cfg_p.get("alerte_prime_achat_max_pct", 3.0)), step=0.5)
        with col2:
            cible_or = st.slider("Cible allocation or (%)", 0, 100, int(cfg_p.get("cible_allocation_or_pct", 70)))
            st.caption(f"Cible argent : {100 - cible_or}%")

        if st.form_submit_button("💾 Sauvegarder"):
            set_config("portfolio", {**cfg_p,
                "alerte_pl_vente_pct": pl_alerte,
                "alerte_prime_achat_max_pct": prime_alerte,
                "cible_allocation_or_pct": cible_or,
                "cible_allocation_argent_pct": 100 - cible_or,
            })
            st.success("✅ Paramètres portefeuille sauvegardés.")

# ---------------------------------------------------------------------------
# Tab 4 : API
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Clé API FRED (taux réels, CPI)")
    cfg_api = get_config("api") or {}

    st.markdown(
        "Obtenez une clé gratuite sur [fred.stlouisfed.org/docs/api/api_key.html]"
        "(https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    with st.form("form_api"):
        fred_key = st.text_input(
            "Clé API FRED",
            value=cfg_api.get("fred_api_key", ""),
            type="password",
            placeholder="Laissez vide pour utiliser pandas-datareader (sans clé)",
        )
        cache_ttl = st.number_input("TTL cache (heures)", value=int(cfg_api.get("cache_ttl_heures", 24)), step=1)

        if st.form_submit_button("💾 Sauvegarder"):
            set_config("api", {**cfg_api, "fred_api_key": fred_key, "cache_ttl_heures": cache_ttl})
            st.success("✅ Configuration API sauvegardée.")

    st.info("Sans clé FRED, les taux réels TIPS seront récupérés via pandas-datareader (peut être plus lent).")

# ---------------------------------------------------------------------------
# Tab 5 : Import / Export
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Export de la configuration complète")
    full_config = load_full_config_from_db()
    json_str = export_config_json(full_config)

    st.download_button(
        "⬇️ Télécharger config.json",
        data=json_str,
        file_name=export_filename("goldsignal_config"),
        mime="application/json",
    )

    st.markdown("---")
    st.subheader("Import d'une configuration")
    uploaded = st.file_uploader("Charger un fichier config.json", type=["json"])
    if uploaded:
        try:
            imported = import_config_json(uploaded.read().decode("utf-8"))
            st.json(imported)
            if st.button("✅ Appliquer cette configuration", type="primary"):
                for key, value in imported.items():
                    if key not in ("_comment", "version", "pieces"):
                        set_config(key, value)
                if "pieces" in imported:
                    for piece in imported["pieces"]:
                        upsert_piece(piece)
                st.success("Configuration importée avec succès.")
                st.rerun()
        except ValueError as exc:
            st.error(f"Fichier invalide : {exc}")

    st.markdown("---")
    st.subheader("Réinitialiser la configuration")
    if st.button("🔄 Réinitialiser aux valeurs par défaut", type="secondary"):
        import json
        from pathlib import Path
        from data.database import seed_default_config
        cfg_path = Path(__file__).parent.parent / "config" / "default_config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                default = json.load(f)
            seed_default_config(default)
            st.success("✅ Configuration réinitialisée aux valeurs par défaut.")
        else:
            st.error("Fichier default_config.json introuvable.")

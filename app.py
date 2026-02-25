"""
app.py — Point d'entrée GoldSignal.

Initialise la base SQLite, charge la config par défaut,
puis configure la navigation multi-pages Streamlit.

Lancement :
    streamlit run app.py
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Résolution du path (pour les imports relatifs)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation base de données (une seule fois par session)
# ---------------------------------------------------------------------------
if "db_initialized" not in st.session_state:
    from data.database import init_db, seed_default_config

    init_db()

    config_path = ROOT / "config" / "default_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            default_cfg = json.load(f)
        seed_default_config(default_cfg)
        logger.info("Config par défaut seedée.")

    st.session_state["db_initialized"] = True

# ---------------------------------------------------------------------------
# Configuration Streamlit globale
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GoldSignal",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GoldSignal v1.0 — Analyse & aide à la décision pour métaux précieux physiques.",
    },
)

# ---------------------------------------------------------------------------
# PWA : injection du manifest + enregistrement du Service Worker
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#f59e0b">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="GoldSignal">
    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
          .then(r => console.log('[PWA] SW enregistré :', r.scope))
          .catch(e => console.warn('[PWA] SW erreur :', e));
      }
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CSS global (thème sombre, typographie, mobile-first)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Typographie globale */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Métriques */
    [data-testid="metric-container"] {
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(148,163,184,0.25);
    }

    /* Boutons primaires */
    .stButton > button[kind="primary"] {
        background: #f59e0b;
        color: #0f172a;
        font-weight: 700;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #d97706;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 8px;
    }

    /* Masquer le "Made with Streamlit" footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation multi-pages (Streamlit ≥ 1.32 — st.Page / st.navigation)
# ---------------------------------------------------------------------------

pages = [
    st.Page(
        "pages/00_accueil.py",
        title="Accueil",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "pages/01_calculateur.py",
        title="Calculateur",
        icon="🧮",
    ),
    st.Page(
        "pages/02_macro.py",
        title="Macro & Technique",
        icon="📊",
    ),
    st.Page(
        "pages/03_predictions.py",
        title="Prédictions ML",
        icon="🤖",
    ),
    st.Page(
        "pages/04_portfolio.py",
        title="Simulateur",
        icon="💰",
    ),
    st.Page(
        "pages/07_backtest.py",
        title="Backtesting P&L",
        icon="📈",
    ),
    st.Page(
        "pages/06_benchmark.py",
        title="Benchmark ML",
        icon="📐",
    ),
    st.Page(
        "pages/05_config.py",
        title="Config",
        icon="⚙️",
    ),
]

# Sidebar : logo + infos spots rapides
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 16px 0 8px'>
          <span style='font-size:2.2em'>🥇</span>
          <h2 style='color:#f59e0b; margin:4px 0 0'>GoldSignal</h2>
          <small style='color:#94a3b8'>Métaux précieux physiques</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Spots rapides (chargés en cache pour la sidebar)
    try:
        from data.fetcher import get_spot_xau_eur, get_spot_xag_eur
        from utils.formatting import fmt_eur

        xau_g = get_spot_xau_eur()
        xag_g = get_spot_xag_eur()

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🥇 Or /g fin", fmt_eur(xau_g) if xau_g else "—")
        with col2:
            st.metric("🥈 Argent /g fin", fmt_eur(xag_g, 4) if xag_g else "—")

        # Napoléon 20F indicatif
        if xau_g:
            nap20_spot = xau_g * 5.806
            st.caption(f"Napoléon 20F spot≈ {fmt_eur(nap20_spot, 0)}")
    except Exception as exc:
        logger.debug("Spots sidebar non disponibles : %s", exc)

    st.markdown("---")

# Lancement de la navigation
pg = st.navigation(pages)
pg.run()

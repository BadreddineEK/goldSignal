"""
database.py — Couche SQLite pour GoldSignal.

Tables gérées :
  - config          : paramètres de l'application (JSON blob par clé)
  - pieces          : catalogue des pièces (poids fin, seuils)
  - price_cache     : cache des séries temporelles yfinance/FRED
  - portfolio       : achats personnels
  - alerts          : historique des alertes déclenchées
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Chemin vers la base SQLite
# Sur Streamlit Community Cloud, data/ est en lecture seule → fallback /tmp
_DATA_DIR = Path(__file__).parent
if not _DATA_DIR.exists() or not os.access(_DATA_DIR, os.W_OK):
    _DB_PATH = Path("/tmp/goldsignal.db")
else:
    _DB_PATH = _DATA_DIR / "goldsignal.db"


# ---------------------------------------------------------------------------
# Connexion
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Retourne une connexion SQLite avec row_factory Row activée."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # meilleure concurrence
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Initialisation des tables
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas encore.

    À appeler au démarrage de l'application (app.py).
    """
    sql_statements = [
        # --- Config clé/valeur (JSON blob) --------------------------------
        """
        CREATE TABLE IF NOT EXISTS config (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL,           -- JSON serialisé
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,

        # --- Catalogue des pièces -----------------------------------------
        """
        CREATE TABLE IF NOT EXISTS pieces (
            id          TEXT PRIMARY KEY,
            nom         TEXT NOT NULL,
            metal       TEXT NOT NULL CHECK(metal IN ('or', 'argent')),
            g_fin       REAL NOT NULL,
            compare_to  TEXT REFERENCES pieces(id),
            actif       INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,

        # --- Seuils prime/spread par pièce --------------------------------
        """
        CREATE TABLE IF NOT EXISTS seuils_pieces (
            piece_id            TEXT NOT NULL REFERENCES pieces(id),
            prime_good_max      REAL NOT NULL DEFAULT 2.0,
            prime_warn_max      REAL NOT NULL DEFAULT 5.0,
            spread_good_max     REAL NOT NULL DEFAULT 2.0,
            spread_warn_max     REAL NOT NULL DEFAULT 4.0,
            score_good_max      REAL NOT NULL DEFAULT 4.0,
            score_warn_max      REAL NOT NULL DEFAULT 8.0,
            updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (piece_id)
        );
        """,

        # --- Cache séries temporelles ------------------------------------
        """
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker      TEXT NOT NULL,
            date_str    TEXT NOT NULL,        -- 'YYYY-MM-DD'
            open_price  REAL,
            high_price  REAL,
            low_price   REAL,
            close_price REAL,
            volume      REAL,
            fetched_at  TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (ticker, date_str)
        );
        """,

        # --- Macro / FRED cache ------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS macro_cache (
            serie       TEXT NOT NULL,        -- ex: 'DFII10', 'CPIAUCSL'
            date_str    TEXT NOT NULL,
            value       REAL,
            fetched_at  TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (serie, date_str)
        );
        """,

        # --- Portefeuille : achats personnels ----------------------------
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_achat  TEXT NOT NULL,            -- 'YYYY-MM-DD'
            piece_id    TEXT NOT NULL REFERENCES pieces(id),
            quantite    REAL NOT NULL DEFAULT 1,
            prix_ask    REAL NOT NULL,             -- € payé à l'achat
            comptoir    TEXT,
            note        TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,

        # --- Alertes déclenchées -----------------------------------------
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT NOT NULL,            -- 'achat' | 'vente' | 'info'
            message     TEXT NOT NULL,
            triggered_at TEXT NOT NULL DEFAULT (datetime('now')),
            acknowledged INTEGER NOT NULL DEFAULT 0
        );
        """,

        # --- Résultats d'entraînement ML (persistance cross-session) ------
        """
        CREATE TABLE IF NOT EXISTS ml_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at       TEXT NOT NULL DEFAULT (datetime('now')),
            horizon      INTEGER NOT NULL DEFAULT 5,
            n_splits     INTEGER NOT NULL DEFAULT 5,
            metrics_json TEXT NOT NULL,   -- JSON : {model: {da_pct, brier, ...}}
            signals_json TEXT NOT NULL,   -- JSON : {model: {date: signal}}
            params_json  TEXT NOT NULL DEFAULT '{}'  -- hyperparamètres
        );
        """,

        # --- Historique des signaux émis -----------------------------------
        """
        CREATE TABLE IF NOT EXISTS signal_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date  TEXT NOT NULL,
            model        TEXT NOT NULL,
            signal       INTEGER NOT NULL,   -- -1 | 0 | 1
            p_haussier   REAL,
            p_neutre     REAL,
            p_baissier   REAL,
            horizon      INTEGER,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(signal_date, model)
        );
        """,
    ]

    with get_connection() as conn:
        for stmt in sql_statements:
            conn.execute(stmt)
        conn.commit()

    logger.info("Base SQLite initialisée : %s", _DB_PATH)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_config(key: str, default=None):
    """Lit une valeur de config depuis SQLite (JSON désérialisé).

    Args:
        key: Clé de configuration.
        default: Valeur retournée si la clé est absente.

    Returns:
        Valeur Python désérialisée ou ``default``.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def set_config(key: str, value) -> None:
    """Sauvegarde une valeur de config dans SQLite (JSON sérialisé).

    Args:
        key: Clé de configuration.
        value: Valeur Python sérialisable en JSON.
    """
    json_val = json.dumps(value, ensure_ascii=False)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO config (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, json_val),
        )
        conn.commit()


def load_full_config_from_db() -> dict:
    """Retourne toute la config stockée sous forme de dict {key: value}."""
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM config").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


# ---------------------------------------------------------------------------
# Pièces helpers
# ---------------------------------------------------------------------------

def get_pieces(actif_only: bool = True) -> list[dict]:
    """Retourne la liste des pièces du catalogue.

    Args:
        actif_only: Si True, retourne uniquement les pièces actives.

    Returns:
        Liste de dictionnaires représentant chaque pièce.
    """
    with get_connection() as conn:
        if actif_only:
            rows = conn.execute(
                "SELECT * FROM pieces WHERE actif = 1 ORDER BY metal, nom"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pieces ORDER BY metal, nom"
            ).fetchall()
    return [dict(r) for r in rows]


def upsert_piece(piece: dict) -> None:
    """Insère ou met à jour une pièce dans le catalogue.

    Args:
        piece: Dict avec clés ``id``, ``nom``, ``metal``, ``g_fin``,
               ``compare_to`` (nullable), ``actif``.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO pieces (id, nom, metal, g_fin, compare_to, actif, updated_at)
            VALUES (:id, :nom, :metal, :g_fin, :compare_to, :actif, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                nom = excluded.nom,
                metal = excluded.metal,
                g_fin = excluded.g_fin,
                compare_to = excluded.compare_to,
                actif = excluded.actif,
                updated_at = excluded.updated_at
            """,
            piece,
        )
        conn.commit()


def get_seuils_piece(piece_id: str) -> dict | None:
    """Retourne les seuils prime/spread pour une pièce donnée."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM seuils_pieces WHERE piece_id = ?", (piece_id,)
        ).fetchone()
    return dict(row) if row else None


def upsert_seuils_piece(seuils: dict) -> None:
    """Insère ou met à jour les seuils pour une pièce.

    Args:
        seuils: Dict avec au moins ``piece_id`` et les champs de seuil.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO seuils_pieces
                (piece_id, prime_good_max, prime_warn_max,
                 spread_good_max, spread_warn_max,
                 score_good_max, score_warn_max, updated_at)
            VALUES
                (:piece_id, :prime_good_max, :prime_warn_max,
                 :spread_good_max, :spread_warn_max,
                 :score_good_max, :score_warn_max, datetime('now'))
            ON CONFLICT(piece_id) DO UPDATE SET
                prime_good_max  = excluded.prime_good_max,
                prime_warn_max  = excluded.prime_warn_max,
                spread_good_max = excluded.spread_good_max,
                spread_warn_max = excluded.spread_warn_max,
                score_good_max  = excluded.score_good_max,
                score_warn_max  = excluded.score_warn_max,
                updated_at      = excluded.updated_at
            """,
            seuils,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Price cache helpers
# ---------------------------------------------------------------------------

def get_cached_prices(ticker: str) -> list[dict]:
    """Retourne toutes les lignes de cache pour un ticker donné.

    Args:
        ticker: Ticker yfinance (ex: 'GC=F').

    Returns:
        Liste de dicts triée par date croissante.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM price_cache WHERE ticker = ? ORDER BY date_str",
            (ticker,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_cache_last_date(ticker: str) -> str | None:
    """Retourne la date la plus récente en cache pour un ticker.

    Args:
        ticker: Ticker yfinance.

    Returns:
        Chaîne 'YYYY-MM-DD' ou None si pas de cache.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(date_str) AS last FROM price_cache WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    return row["last"] if row else None


def insert_prices_bulk(ticker: str, rows: list[dict]) -> None:
    """Insère en masse des prix dans le cache (ignore les doublons).

    Args:
        ticker: Ticker yfinance.
        rows: Liste de dicts avec clés ``date_str``, ``open_price``,
              ``high_price``, ``low_price``, ``close_price``, ``volume``.
    """
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO price_cache
                (ticker, date_str, open_price, high_price, low_price, close_price, volume)
            VALUES
                (:ticker, :date_str, :open, :high, :low, :close, :volume)
            """,
            [{"ticker": ticker, **r} for r in rows],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Macro / FRED cache helpers
# ---------------------------------------------------------------------------

def get_cached_macro(serie: str) -> list[dict]:
    """Retourne les données macro en cache pour une série FRED.

    Args:
        serie: Identifiant FRED (ex: 'DFII10', 'CPIAUCSL').

    Returns:
        Liste de dicts triée par date.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM macro_cache WHERE serie = ? ORDER BY date_str",
            (serie,),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_macro_bulk(serie: str, rows: list[dict]) -> None:
    """Insère en masse des séries macro dans le cache.

    Args:
        serie: Identifiant FRED.
        rows: Liste de dicts avec clés ``date_str``, ``value``.
    """
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO macro_cache (serie, date_str, value)
            VALUES (:serie, :date_str, :value)
            """,
            [{"serie": serie, **r} for r in rows],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def add_achat(date_achat: str, piece_id: str, quantite: float,
              prix_ask: float, comptoir: str = "", note: str = "") -> int:
    """Enregistre un achat dans le portefeuille.

    Args:
        date_achat: Date de l'achat au format 'YYYY-MM-DD'.
        piece_id:   Identifiant de la pièce.
        quantite:   Nombre de pièces achetées.
        prix_ask:   Prix Ask payé en €.
        comptoir:   Nom du comptoir (optionnel).
        note:       Commentaire libre (optionnel).

    Returns:
        rowid de la ligne insérée.
    """
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO portfolio (date_achat, piece_id, quantite, prix_ask, comptoir, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (date_achat, piece_id, quantite, prix_ask, comptoir, note),
        )
        conn.commit()
        return cur.lastrowid


def get_portfolio() -> list[dict]:
    """Retourne tous les achats du portefeuille, joints avec les pièces.

    Returns:
        Liste de dicts triée par date décroissante.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.*, pc.nom AS piece_nom, pc.metal, pc.g_fin
            FROM portfolio p
            JOIN pieces pc ON p.piece_id = pc.id
            ORDER BY p.date_achat DESC
            """,
        ).fetchall()
    return [dict(r) for r in rows]


def delete_achat(achat_id: int) -> None:
    """Supprime un achat du portefeuille.

    Args:
        achat_id: ID de la ligne portfolio.
    """
    with get_connection() as conn:
        conn.execute("DELETE FROM portfolio WHERE id = ?", (achat_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Alerts helpers
# ---------------------------------------------------------------------------

def add_alert(type_: str, message: str) -> None:
    """Enregistre une alerte.

    Args:
        type_: 'achat' | 'vente' | 'info'.
        message: Texte de l'alerte.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO alerts (type, message) VALUES (?, ?)",
            (type_, message),
        )
        conn.commit()


def get_unacknowledged_alerts() -> list[dict]:
    """Retourne les alertes non acquittées."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY triggered_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int) -> None:
    """Marque une alerte comme acquittée.

    Args:
        alert_id: ID de l'alerte.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# ML Runs — persistance des résultats d'entraînement
# ---------------------------------------------------------------------------

def save_ml_run(
    metrics: dict,
    signals: dict,
    horizon: int = 5,
    n_splits: int = 5,
    params: dict | None = None,
) -> int:
    """Sauvegarde les métriques et signaux d'un run ML en base.

    Args:
        metrics: {model_name: {da_pct, brier, log_loss, ...}}
        signals: {model_name: {date_str: signal_int}}
        horizon: Horizon utilisé.
        n_splits: Nombre de folds.
        params:  Hyperparamètres.

    Returns:
        run_id (INTEGER PRIMARY KEY)
    """
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO ml_runs (horizon, n_splits, metrics_json, signals_json, params_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                horizon,
                n_splits,
                json.dumps(metrics, ensure_ascii=False, default=str),
                json.dumps(signals, ensure_ascii=False, default=str),
                json.dumps(params or {}, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_latest_ml_run() -> dict | None:
    """Retourne le dernier run ML sauvegardé.

    Returns:
        Dict avec clés id, run_at, horizon, n_splits, metrics, signals, params
        ou None si aucun run exist.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM ml_runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return {
        "id":       row["id"],
        "run_at":   row["run_at"],
        "horizon":  row["horizon"],
        "n_splits": row["n_splits"],
        "metrics":  json.loads(row["metrics_json"]),
        "signals":  json.loads(row["signals_json"]),
        "params":   json.loads(row["params_json"]),
    }


def list_ml_runs(limit: int = 20) -> list[dict]:
    """Retourne les N derniers runs ML (métadonnées sans les signaux)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, run_at, horizon, n_splits, params_json "
            "FROM ml_runs ORDER BY run_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r["id"], "run_at": r["run_at"], "horizon": r["horizon"],
             "n_splits": r["n_splits"], "params": json.loads(r["params_json"])}
            for r in rows]


# ---------------------------------------------------------------------------
# Signal history
# ---------------------------------------------------------------------------

def save_signal(
    signal_date: str,
    model: str,
    signal: int,
    p_haussier: float | None = None,
    p_neutre: float | None = None,
    p_baissier: float | None = None,
    horizon: int | None = None,
) -> None:
    """Enregistre un signal temps réel dans l'historique.

    Args:
        signal_date: Date du signal 'YYYY-MM-DD'.
        model:       Nom du modèle ('RF', 'XGB', 'LSTM', 'Hybride').
        signal:      -1 | 0 | 1.
        p_haussier:  Probabilité hausse.
        p_neutre:    Probabilité neutre.
        p_baissier:  Probabilité baisse.
        horizon:     Horizon jours.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO signal_history
                (signal_date, model, signal, p_haussier, p_neutre, p_baissier, horizon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_date, model) DO UPDATE SET
                signal     = excluded.signal,
                p_haussier = excluded.p_haussier,
                p_neutre   = excluded.p_neutre,
                p_baissier = excluded.p_baissier,
                horizon    = excluded.horizon,
                created_at = datetime('now')
            """,
            (signal_date, model, signal, p_haussier, p_neutre, p_baissier, horizon),
        )
        conn.commit()


def get_signal_history(model: str | None = None, limit: int = 90) -> list[dict]:
    """Retourne l'historique des signaux.

    Args:
        model: Filtrer par modèle (None = tous).
        limit: Nombre max de lignes.

    Returns:
        Liste de dicts triée par date décroissante.
    """
    with get_connection() as conn:
        if model:
            rows = conn.execute(
                "SELECT * FROM signal_history WHERE model = ? "
                "ORDER BY signal_date DESC LIMIT ?",
                (model, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signal_history ORDER BY signal_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Seed initial data
# ---------------------------------------------------------------------------

def seed_default_config(default_config: dict) -> None:
    """Charge la config par défaut dans SQLite si elle n'existe pas encore.

    Ne remplace PAS les valeurs déjà présentes en base.

    Args:
        default_config: Dictionnaire lu depuis ``default_config.json``.
    """
    # Pièces
    for piece in default_config.get("pieces", []):
        existing = get_pieces(actif_only=False)
        existing_ids = {p["id"] for p in existing}
        if piece["id"] not in existing_ids:
            upsert_piece(piece)
            metal = piece["metal"]
            seuils_metal = default_config.get("seuils", {}).get(metal, {})
            upsert_seuils_piece({"piece_id": piece["id"], **seuils_metal})

    # Config globale (sections : macro, modeles, portfolio, api, tickers)
    for section in ["macro", "modeles", "portfolio", "api", "tickers_yfinance"]:
        if get_config(section) is None and section in default_config:
            set_config(section, default_config[section])

    logger.info("Seed config initiale effectué.")

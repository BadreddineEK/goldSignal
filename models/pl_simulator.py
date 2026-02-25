"""
pl_simulator.py — Simulateur P&L sur signaux de trading GoldSignal.

Protocole :
  Les signaux proviennent uniquement des prédictions out-of-sample (walk-forward).
  Zéro look-ahead : un signal au jour J est utilisé pour la période J→J+horizon.

Stratégie simulée :
  - Signal haussier (+1)  → position LONGUE (on achète/détient de l'or)
  - Signal baissier (-1)  → position NEUTRE (on sort ou on n'achète pas)
  - Signal neutre (0)     → position NEUTRE

Métriques calculées :
  - Courbe de capitalisation (equity curve)
  - Rendement total, annualisé, vs buy-and-hold
  - Sharpe ratio (annualisé, rf=0)
  - Sortino ratio
  - Maximum drawdown (MDD)
  - Calmar ratio = rendement annualisé / MDD
  - Win rate, profit factor, avg gain, avg loss
  - Nombre de trades, durée moyenne
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_YEAR = 252


# ---------------------------------------------------------------------------
# Equity curve engine
# ---------------------------------------------------------------------------

def simulate_strategy(
    signals: pd.Series,
    log_returns: pd.Series,
    initial_capital: float = 10_000.0,
    cost_per_trade_pct: float = 0.001,  # 0.1% frais aller/retour
) -> pd.DataFrame:
    """Simule une stratégie long-only basée sur les signaux ML.

    Aligne les signaux sur les log-returns, applique les frais de transaction
    uniquement lors des changements de position.

    Args:
        signals:         pd.Series de signaux {-1, 0, +1}, indexée par date.
        log_returns:     pd.Series de log-returns réels, indexée par date.
        initial_capital: Capital de départ en €.
        cost_per_trade_pct: Frais aller-retour en % du capital par trade.

    Returns:
        DataFrame avec colonnes :
          signal, log_return, position, strat_return, equity, drawdown_pct
    """
    # Alignement sur les dates communes
    common_idx = signals.index.intersection(log_returns.index)
    if common_idx.empty:
        logger.warning("Aucune date commune entre signaux et log-returns.")
        return pd.DataFrame()

    sig = signals.reindex(common_idx).fillna(0).astype(int)
    lr  = log_returns.reindex(common_idx).fillna(0.0)

    # Position longue uniquement sur signal haussier
    position = (sig == 1).astype(float)

    # Frais : appliqués à chaque changement de position (entrée/sortie)
    pos_change = position.diff().abs().fillna(0)
    transaction_costs = pos_change * cost_per_trade_pct

    # Rendements stratégie = position * log_return - frais
    strat_return = position * lr - transaction_costs

    # Equity curve
    equity = initial_capital * np.exp(strat_return.cumsum())

    # Buy & Hold
    bh_equity = initial_capital * np.exp(lr.cumsum())

    # Drawdown
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100.0

    df = pd.DataFrame({
        "signal":       sig,
        "log_return":   lr,
        "position":     position,
        "strat_return": strat_return,
        "equity":       equity,
        "bh_equity":    bh_equity,
        "drawdown_pct": drawdown_pct,
    }, index=common_idx)

    return df


# ---------------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------------

def compute_pl_metrics(df: pd.DataFrame, horizon: int = 5) -> dict:
    """Calcule toutes les métriques P&L à partir du DataFrame de simulation.

    Args:
        df:      DataFrame retourné par simulate_strategy().
        horizon: Horizon de prédiction (jours). Utilisé pour comptabiliser les trades.

    Returns:
        Dict complet de métriques.
    """
    if df.empty or "strat_return" not in df.columns:
        return {}

    r = df["strat_return"]
    n = len(r)

    # --- Rendements ---
    total_return_pct    = float((df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100)
    bh_total_return_pct = float((df["bh_equity"].iloc[-1] / df["bh_equity"].iloc[0] - 1) * 100)
    years = n / TRADING_DAYS_YEAR
    annualized_return   = float(((df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (1 / max(years, 0.1)) - 1) * 100)
    bh_annualized       = float(((df["bh_equity"].iloc[-1] / df["bh_equity"].iloc[0]) ** (1 / max(years, 0.1)) - 1) * 100)
    alpha               = annualized_return - bh_annualized

    # --- Volatilité & Sharpe ---
    daily_vol = float(r.std())
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_YEAR) * 100
    sharpe = (annualized_return / annual_vol) if annual_vol > 0 else 0.0

    # Sortino (downside only)
    downside = r[r < 0]
    downside_vol = float(downside.std()) * np.sqrt(TRADING_DAYS_YEAR) * 100 if len(downside) > 1 else annual_vol
    sortino = (annualized_return / downside_vol) if downside_vol > 0 else 0.0

    # --- Drawdown ---
    max_drawdown = float(df["drawdown_pct"].min())  # valeur négative
    calmar = (annualized_return / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

    # --- Trades (un "trade" = séquence consécutive position=1 d'horizon jours) ---
    pos = df["position"]
    entries  = ((pos == 1) & (pos.shift(1) != 1)).sum()
    n_trades = int(entries)

    # Win rate sur les trades actifs
    active_returns = r[df["position"] == 1]
    # Grouper par blocs consécutifs
    trade_groups = (pos != pos.shift()).cumsum()
    trade_pl = []
    for gid, grp in df[df["position"] == 1].groupby(trade_groups[df["position"] == 1]):
        trade_pl.append(float(grp["strat_return"].sum()))

    win_rate        = float(np.mean([x > 0 for x in trade_pl])) * 100 if trade_pl else 0.0
    avg_gain        = float(np.mean([x for x in trade_pl if x > 0]) * 100) if any(x > 0 for x in trade_pl) else 0.0
    avg_loss        = float(np.mean([x for x in trade_pl if x < 0]) * 100) if any(x < 0 for x in trade_pl) else 0.0
    profit_factor   = (abs(avg_gain) / abs(avg_loss)) if avg_loss < 0 else float("inf")
    avg_duration    = float(len(df[df["position"] == 1]) / max(n_trades, 1))

    # --- Exposition ---
    exposure_pct = float((df["position"] == 1).mean() * 100)

    return {
        # rendements
        "total_return_pct":     round(total_return_pct, 2),
        "annualized_return_pct":round(annualized_return, 2),
        "bh_total_return_pct":  round(bh_total_return_pct, 2),
        "bh_annualized_pct":    round(bh_annualized, 2),
        "alpha_pct":            round(alpha, 2),
        # risque
        "annual_vol_pct":       round(annual_vol, 2),
        "sharpe":               round(sharpe, 3),
        "sortino":              round(sortino, 3),
        "max_drawdown_pct":     round(max_drawdown, 2),
        "calmar":               round(calmar, 3),
        # trades
        "n_trades":             n_trades,
        "win_rate_pct":         round(win_rate, 1),
        "avg_gain_pct":         round(avg_gain, 3),
        "avg_loss_pct":         round(avg_loss, 3),
        "profit_factor":        round(profit_factor, 2) if profit_factor != float("inf") else 999,
        "avg_duration_days":    round(avg_duration, 1),
        "exposure_pct":         round(exposure_pct, 1),
        # taille
        "n_observations":       n,
    }


# ---------------------------------------------------------------------------
# Extraction des signaux depuis les résultats ML
# ---------------------------------------------------------------------------

def extract_signals_from_ml(ml_results: dict, model_key: str = "rf") -> pd.Series:
    """Extrait les prédictions out-of-sample comme pd.Series de signaux {-1,0,+1}.

    Args:
        ml_results: Résultats de run_all_ml_models() (session_state["ml_results"]).
        model_key: 'rf' | 'xgb'.

    Returns:
        pd.Series indexée par date avec valeurs {-1, 0, 1}.
    """
    model_data = ml_results.get(model_key, {})
    preds = model_data.get("predictions")  # dict {date: label} ou pd.Series
    if preds is None:
        return pd.Series(dtype=int)

    if isinstance(preds, dict):
        return pd.Series(preds, dtype=int).sort_index()
    if isinstance(preds, pd.Series):
        return preds.sort_index().astype(int)
    return pd.Series(dtype=int)


def extract_signals_from_lstm(lstm_result: dict) -> pd.Series:
    """Extrait les signaux depuis les résultats LSTM."""
    preds = lstm_result.get("predictions")
    if preds is None:
        return pd.Series(dtype=int)
    if isinstance(preds, pd.Series):
        return preds.sort_index().astype(int)
    return pd.Series(dtype=int)


def extract_signals_from_hybrid(hybrid_result: dict) -> pd.Series:
    """Extrait les signaux depuis les résultats hybrides."""
    preds = hybrid_result.get("predictions")
    if preds is None:
        return pd.Series(dtype=int)
    if isinstance(preds, pd.Series):
        return preds.sort_index().astype(int)
    return pd.Series(dtype=int)


# ---------------------------------------------------------------------------
# Utilitaire : construire le tableau de comparaison des stratégies
# ---------------------------------------------------------------------------

def build_strategy_comparison(
    results_by_model: dict[str, dict],
) -> pd.DataFrame:
    """Construit un tableau comparatif des métriques P&L pour tous les modèles.

    Args:
        results_by_model: {nom_modèle: métriques_dict (compute_pl_metrics output)}.

    Returns:
        DataFrame formaté.
    """
    rows = []
    for model, m in results_by_model.items():
        if not m:
            continue
        rows.append({
            "Modèle":               model,
            "Rdt total (%)":        m.get("total_return_pct"),
            "Rdt annualisé (%)":    m.get("annualized_return_pct"),
            "B&H annualisé (%)":    m.get("bh_annualized_pct"),
            "Alpha (%)":            m.get("alpha_pct"),
            "Sharpe":               m.get("sharpe"),
            "Sortino":              m.get("sortino"),
            "Max Drawdown (%)":     m.get("max_drawdown_pct"),
            "Calmar":               m.get("calmar"),
            "Win Rate (%)":         m.get("win_rate_pct"),
            "Profit Factor":        m.get("profit_factor"),
            "# Trades":             m.get("n_trades"),
            "Expo (%)":             m.get("exposure_pct"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Sharpe", ascending=False)
    return df

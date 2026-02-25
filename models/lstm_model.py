"""
lstm_model.py — LSTM PyTorch pour la prédiction de tendance sur l'or.

Architecture :
  - TimeSeriesDataset    : sliding window sur données normalisées (MinMaxScaler par fold)
  - LSTMClassifier       : LSTM multi-couches + mécanisme d'attention dot-product + FC
  - EarlyStopping        : patience sur validation loss
  - WalkForward training : expansion stricte, normalisation dans le fold (zero leakage)

Sortie :
  - Probas ternaires {hausier, neutre, baissier}  ←→ compatible signal_generator.py
  - Test de Diebold-Mariano vs Random Walk
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# PyTorch optionnel — désactivé gracieusement sur Streamlit Cloud si absent
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = nn = F = DataLoader = Dataset = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-paramètres par défaut
# ---------------------------------------------------------------------------
DEFAULT_SEQ_LEN      = 30     # fenêtre temporelle (jours)
DEFAULT_HIDDEN_SIZE  = 64     # neurones LSTM par couche
DEFAULT_NUM_LAYERS   = 2      # couches LSTM empilées
DEFAULT_DROPOUT      = 0.3    # dropout entre couches
DEFAULT_LR           = 1e-3
DEFAULT_EPOCHS       = 100
DEFAULT_BATCH        = 32
DEFAULT_PATIENCE     = 10     # early stopping
DEFAULT_WEIGHT_DECAY = 1e-4   # L2 régularisation


# ---------------------------------------------------------------------------
# Dataset PyTorch — sliding window
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Fenêtre glissante sur une matrice de features + étiquettes.

    Args:
        X: ndarray (T, F) — features normalisées.
        y: ndarray (T,)   — labels entiers {0, 1, 2}.
        seq_len: Longueur de la séquence.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = DEFAULT_SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]        # (seq_len, F)
        label = self.y[idx + self.seq_len]              # scalaire
        return x_seq, label


# ---------------------------------------------------------------------------
# Mécanisme d'attention dot-product
# ---------------------------------------------------------------------------
class DotProductAttention(nn.Module):
    """Attention scalée sur les sorties LSTM (Bahdanau-light).

    Calcule un contexte pondéré sur la séquence, permettant au modèle
    de se concentrer sur les timesteps les plus informatifs.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key   = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context : (batch, hidden_size)
            weights : (batch, seq_len) — poids d'attention (interprétable)
        """
        q = self.query(lstm_out[:, -1:, :])     # (batch, 1, H)
        k = self.key(lstm_out)                   # (batch, seq, H)
        scores = torch.bmm(q, k.transpose(1, 2)) / (lstm_out.size(-1) ** 0.5)
        weights = F.softmax(scores.squeeze(1), dim=-1)   # (batch, seq)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, H)
        return context, weights


# ---------------------------------------------------------------------------
# Architecture LSTM + Attention + FC
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """Classifieur LSTM multi-couches avec attention et sortie ternaire.

    Architecture :
      LSTM(seq_len→hidden) × num_layers
      → DotProductAttention
      → LayerNorm
      → Dropout
      → FC(hidden → n_classes)
      → Softmax (en inférence)

    Args:
        input_size:  Nombre de features.
        hidden_size: Unités LSTM par couche.
        num_layers:  Couches LSTM empilées.
        n_classes:   Nombre de classes (3 : haussier/neutre/baissier).
        dropout:     Taux de dropout (entre couches, si num_layers > 1).
        bidirectional: LSTM bidirectionnel (double la capacité, expérimentale).
    """

    def __init__(
        self,
        input_size:   int,
        hidden_size:  int   = DEFAULT_HIDDEN_SIZE,
        num_layers:   int   = DEFAULT_NUM_LAYERS,
        n_classes:    int   = 3,
        dropout:      float = DEFAULT_DROPOUT,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        self.attention  = DotProductAttention(hidden_size * self.directions)
        self.layer_norm = nn.LayerNorm(hidden_size * self.directions)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size * self.directions, n_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits       : (batch, n_classes)
            attn_weights : (batch, seq_len) — interprétabilité
        """
        lstm_out, _ = self.lstm(x)    # (batch, seq, H * dirs)
        context, attn_w = self.attention(lstm_out)
        context = self.layer_norm(context)
        context = self.dropout(context)
        logits  = self.fc(context)
        return logits, attn_w


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Arrête l'entraînement si la val_loss ne s'améliore plus."""

    def __init__(self, patience: int = DEFAULT_PATIENCE, delta: float = 1e-5):
        self.patience    = patience
        self.delta       = delta
        self.best_loss   = np.inf
        self.counter     = 0
        self.best_state  = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Entraînement d'un fold
# ---------------------------------------------------------------------------
def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    seq_len:     int   = DEFAULT_SEQ_LEN,
    hidden_size: int   = DEFAULT_HIDDEN_SIZE,
    num_layers:  int   = DEFAULT_NUM_LAYERS,
    dropout:     float = DEFAULT_DROPOUT,
    lr:          float = DEFAULT_LR,
    epochs:      int   = DEFAULT_EPOCHS,
    batch_size:  int   = DEFAULT_BATCH,
    patience:    int   = DEFAULT_PATIENCE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    bidirectional: bool = False,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[LSTMClassifier, list[float], list[float], MinMaxScaler]:
    """Entraîne un LSTMClassifier sur un fold walk-forward.

    Normalisation MinMaxScaler ajustée **uniquement** sur X_train (zero leakage).

    Returns:
        model       : LSTMClassifier entraîné (meilleur état par early stopping)
        train_losses: Historique des pertes train
        val_losses  : Historique des pertes val
        scaler      : MinMaxScaler ajusté sur train (à appliquer sur test)
    """
    # --- Normalisation (uniquement sur train) ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    ds_train = TimeSeriesDataset(X_train_s, y_train, seq_len)
    ds_val   = TimeSeriesDataset(X_val_s,   y_val,   seq_len)

    if len(ds_train) == 0 or len(ds_val) == 0:
        raise ValueError("Fold trop petit pour la longueur de séquence choisie.")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    model = LSTMClassifier(
        input_size   = n_features,
        hidden_size  = hidden_size,
        num_layers   = num_layers,
        n_classes    = 3,
        dropout      = dropout,
        bidirectional = bidirectional,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # --- train ---
        model.train()
        t_loss = 0.0
        for xb, yb in dl_train:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(len(dl_train), 1)

        # --- val ---
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                logits, _ = model(xb)
                v_loss += criterion(logits, yb).item()
        v_loss /= max(len(dl_val), 1)

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if stopper(v_loss, model):
            logger.debug("[LSTM] Early stop epoch %d | val_loss=%.4f", epoch, v_loss)
            break

    stopper.restore_best(model)
    return model, train_losses, val_losses, scaler


# ---------------------------------------------------------------------------
# Walk-forward complet
# ---------------------------------------------------------------------------
def run_lstm_walkforward(
    X:              pd.DataFrame,
    log_returns:    pd.Series,
    horizon:        int   = 5,
    seuil_pct:      float = 0.003,
    n_splits:       int   = 5,
    min_train_size: int   = 400,
    test_size:      int   = 60,
    seq_len:        int   = DEFAULT_SEQ_LEN,
    hidden_size:    int   = DEFAULT_HIDDEN_SIZE,
    num_layers:     int   = DEFAULT_NUM_LAYERS,
    dropout:        float = DEFAULT_DROPOUT,
    lr:             float = DEFAULT_LR,
    epochs:         int   = DEFAULT_EPOCHS,
    batch_size:     int   = DEFAULT_BATCH,
    patience:       int   = DEFAULT_PATIENCE,
    bidirectional:  bool  = False,
) -> dict:
    """Protocole walk-forward strict pour LSTMClassifier.

    Chaque fold :
      1. MinMaxScaler ajusté **uniquement** sur train → transforme val + évalue.
      2. Aucune information future ne filtre vers le passé.

    Args:
        X:              Features (T, F).
        log_returns:    Log-rendements (T,) — sert à construire la cible.
        horizon:        Horizon de prédiction.
        seuil_pct:      Seuil ternaire.
        n_splits:       Nombre de folds.
        min_train_size: Taille minimale du train.
        test_size:      Taille de la fenêtre de test.
        seq_len:        Longueur de la séquence LSTM.
        hidden_size:    Unités LSTM.
        num_layers:     Couches LSTM.
        dropout:        Dropout rate.
        lr:             Learning rate Adam.
        epochs:         Epochs max.
        batch_size:     Taille du mini-batch.
        patience:       Patience early stopping.
        bidirectional:  LSTM bidirectionnel.

    Returns:
        Dict avec metrics, predictions, actuals, learning_curves, attention_weights.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch n'est pas installé dans cet environnement. "
            "Le LSTM est désactivé sur Streamlit Community Cloud (mémoire insuffisante). "
            "Utilisez RF ou XGBoost."
        )

    from models.ml_models import prepare_classification_target, align_features_target
    from models.backtester import WalkForwardCV, compute_metrics

    # Cible ternaire
    y = prepare_classification_target(log_returns, horizon=horizon, seuil_pct=seuil_pct)
    X_al, y_al = align_features_target(X, y)

    le = LabelEncoder()
    le.fit([-1, 0, 1])

    cv = WalkForwardCV(n_splits=n_splits, min_train_size=min_train_size, test_size=test_size)
    splits = cv.split(X_al)

    if not splits:
        raise ValueError("Pas assez de données pour le walk-forward LSTM.")

    all_preds   = []
    all_actuals = []
    all_probas  = []
    learning_curves = []
    attn_history    = []
    da_per_fold     = []
    last_model      = None
    last_scaler     = None

    logger.info("[LSTM] Démarrage walk-forward — %d folds | horizon=%dj", len(splits), horizon)

    X_np = X_al.values.astype(np.float32)
    y_np = le.transform(y_al.values)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info("[LSTM] Fold %d/%d — train=%d, test=%d",
                    fold_idx + 1, len(splits), len(train_idx), len(test_idx))

        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        # Poids de classe pour rééquilibrer (or : données souvent déséquilibrées)
        counts = np.bincount(y_tr, minlength=3)
        counts = np.maximum(counts, 1)
        cw = torch.tensor(len(y_tr) / (3.0 * counts), dtype=torch.float32)

        # Split val interne (10% du train)
        val_split = max(int(len(X_tr) * 0.1), seq_len + 1)
        X_tr_fold, X_val_fold = X_tr[:-val_split], X_tr[-val_split:]
        y_tr_fold, y_val_fold = y_tr[:-val_split], y_tr[-val_split:]

        try:
            model, t_losses, v_losses, scaler = _train_fold(
                X_tr_fold, y_tr_fold, X_val_fold, y_val_fold,
                seq_len=seq_len, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout,
                lr=lr, epochs=epochs, batch_size=batch_size,
                patience=patience, class_weights=cw,
                bidirectional=bidirectional,
            )
            learning_curves.append({"train": t_losses, "val": v_losses})
        except Exception as exc:
            logger.warning("[LSTM] Fold %d — erreur entraînement : %s", fold_idx + 1, exc)
            continue

        # Prédictions sur le test (fenêtre glissante)
        X_te_s = scaler.transform(X_te)
        fold_preds, fold_probas, fold_attn = _predict_sequence(model, X_te_s, seq_len)

        if len(fold_preds) == 0:
            continue

        # Aligner avec y_te
        offset = seq_len
        y_te_aligned = y_te[offset : offset + len(fold_preds)]
        idx_te = y_al.index[test_idx][offset : offset + len(fold_preds)]

        preds_orig   = le.inverse_transform(fold_preds)
        actuals_orig = le.inverse_transform(y_te_aligned)

        da = np.mean(preds_orig == actuals_orig) * 100
        da_per_fold.append(da)
        logger.info("[LSTM] Fold %d — DA=%.1f%%", fold_idx + 1, da)

        all_preds.append(pd.Series(preds_orig, index=idx_te))
        all_actuals.append(pd.Series(actuals_orig, index=idx_te))
        all_probas.append(fold_probas)
        attn_history.append(fold_attn)

        last_model  = model
        last_scaler = scaler

    if not all_preds:
        raise RuntimeError("[LSTM] Aucun fold valide.")

    preds_series   = pd.concat(all_preds)
    actuals_series = pd.concat(all_actuals)
    all_probas_arr = np.vstack(all_probas)

    # Métriques globales
    da_global = np.mean(preds_series.values == actuals_series.values) * 100
    metrics = {
        "da_pct":    round(da_global, 2),
        "da_folds":  [round(d, 2) for d in da_per_fold],
        "n_samples": len(preds_series),
        "n_folds":   len(da_per_fold),
        "rmse": float(np.sqrt(np.mean((preds_series.values - actuals_series.values) ** 2))),
        "mae":  float(np.mean(np.abs(preds_series.values - actuals_series.values))),
    }

    logger.info("[LSTM] Global — DA=%.1f%% | n=%d", da_global, len(preds_series))

    return {
        "metrics":         metrics,
        "predictions":     preds_series,
        "actuals":         actuals_series,
        "probabilities":   all_probas_arr,   # (N, 3) [baissier, neutre, haussier]
        "learning_curves": learning_curves,
        "attention":       attn_history,
        "da_per_fold":     da_per_fold,
        "last_model":      last_model,
        "last_scaler":     last_scaler,
        "label_encoder":   le,
        "feature_names":   list(X_al.columns),
    }


def _predict_sequence(
    model: LSTMClassifier,
    X_scaled: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prédictions glissantes sur une séquence normalisée.

    Returns:
        preds   : (N,) entiers encodés
        probas  : (N, 3) probabilités softmax
        attns   : (N, seq_len) poids d'attention moyens
    """
    model.eval()
    preds, probas_list, attn_list = [], [], []

    with torch.no_grad():
        for i in range(len(X_scaled) - seq_len):
            x = torch.tensor(
                X_scaled[i : i + seq_len][np.newaxis, :, :], dtype=torch.float32
            )
            logits, attn = model(x)
            prob = F.softmax(logits, dim=-1).squeeze(0).numpy()
            pred = int(np.argmax(prob))
            preds.append(pred)
            probas_list.append(prob)
            attn_list.append(attn.squeeze(0).numpy())

    return (
        np.array(preds, dtype=int),
        np.array(probas_list),
        np.array(attn_list),
    )


# ---------------------------------------------------------------------------
# Inférence temps réel
# ---------------------------------------------------------------------------
def get_latest_lstm_signal(
    model:   LSTMClassifier,
    scaler:  MinMaxScaler,
    X_recent: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    le:      Optional[LabelEncoder] = None,
) -> dict:
    """Signal temps réel à partir des seq_len dernières observations.

    Returns:
        Dict compatible avec signal_generator.probabilities_to_signal().
    """
    if len(X_recent) < seq_len:
        return {}

    X_s = scaler.transform(X_recent.values[-seq_len:].astype(np.float32).reshape(-1, X_recent.shape[1]))
    x = torch.tensor(X_s[np.newaxis, :, :], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits, attn_w = model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    # Classes : 0=baissier, 1=neutre, 2=haussier (encodées par LabelEncoder)
    # LabelEncoder de {-1,0,1} : -1→0, 0→1, 1→2
    p_bais, p_neutre, p_haussier = float(probs[0]), float(probs[1]), float(probs[2])

    from models.signal_generator import probabilities_to_signal
    signal = probabilities_to_signal(p_haussier, p_neutre, p_bais)
    signal["attention_weights"] = attn_w.squeeze(0).numpy().tolist()
    signal["probabilities_raw"] = {"haussier": p_haussier, "neutre": p_neutre, "baissier": p_bais}
    return signal


# ---------------------------------------------------------------------------
# Test de Diebold-Mariano (comparaison statistique de deux séries de prévisions)
# ---------------------------------------------------------------------------
def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h:  int = 1,
) -> dict:
    """Test DM : H0 = les deux modèles ont la même précision prévisionnelle.

    Une p-value < 0.05 signifie que la différence est statistiquement significative.

    Args:
        e1: Erreurs de prévision modèle 1 (ex: |prédits - réels|).
        e2: Erreurs de prévision modèle 2.
        h:  Horizon (correction Harvey-Newbold-Leybourne).

    Returns:
        Dict avec dm_stat, p_value, conclusion.
    """
    import scipy.stats as st

    d = (e1 ** 2) - (e2 ** 2)
    T = len(d)
    if T < 3:
        return {"dm_stat": np.nan, "p_value": np.nan, "conclusion": "Données insuffisantes"}

    d_mean = np.mean(d)
    # Variance long-run (estimateur de Newey-West, lag = h-1)
    gamma0 = np.var(d, ddof=1)
    gammas = [np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)) for k in range(1, h)]
    lr_var = gamma0 + 2 * sum(gammas) if gammas else gamma0
    lr_var = max(lr_var, 1e-12)

    dm_stat = d_mean / np.sqrt(lr_var / T)

    # Correction HNL pour petits échantillons
    correction = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** 0.5
    dm_stat_adj = dm_stat / max(correction, 1e-6)

    p_value = 2 * (1 - st.norm.cdf(abs(dm_stat_adj)))

    if p_value < 0.05:
        better = "Modèle 1" if dm_stat_adj < 0 else "Modèle 2"
        conclusion = f"Différence significative (p={p_value:.3f}) — {better} meilleur"
    else:
        conclusion = f"Différence non significative (p={p_value:.3f})"

    return {
        "dm_stat":   round(float(dm_stat_adj), 4),
        "p_value":   round(float(p_value), 4),
        "conclusion": conclusion,
    }

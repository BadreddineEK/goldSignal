# GoldSignal models package — implémentation Phase 2/3
# Phase 2 : RandomForest, XGBoost, ARIMA, walk-forward
# Phase 3 : LSTM PyTorch (attention), Stacking hybride (méta-apprenant LogReg)

try:
    import torch as _torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "TORCH_AVAILABLE",
    "backtester",
    "baseline",
    "ml_models",
    "signal_generator",
    "lstm_model",
    "hybrid_model",
    "pl_simulator",
]

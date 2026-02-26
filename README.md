# 🥇 GoldSignal

> **Analyse & aide à la décision pour métaux précieux physiques**
> Application Streamlit multi-pages avec ML, backtesting, simulateur et contexte macro — déployée en production.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://goldsignaltime.streamlit.app/)

🔗 **Live demo : [goldsignaltime.streamlit.app](https://goldsignaltime.streamlit.app/)**

---

## 📌 À propos

GoldSignal est un outil complet pour l'investisseur en **métaux précieux physiques** (or, argent, pièces numismatiques). Il combine trois dimensions :

1. **🧮 Évaluation terrain** — Est-ce que le prix affiché par un comptoir est juste ? Calcul prime, spread, score de qualité en temps réel.
2. **📊 Contexte macro & technique** — Dollar, taux réels, VIX, saisonnalité, corrélations. Comprendre les forces qui font bouger les cours.
3. **🤖 Intelligence artificielle** — Prédictions de tendance à 5/15/30 jours via Random Forest, XGBoost, LSTM et un méta-apprenant hybride entraîné en **walk-forward strict** (zéro data leakage).

> ⚠️ GoldSignal est un outil pédagogique et d'aide à la réflexion. Il ne constitue pas un conseil en investissement.

---

## 🚀 Fonctionnalités

| Page | Description |
|------|-------------|
| 🏠 **Accueil** | Cours spot or/argent en temps réel, ratio, Napoléon 20F, signal ML actuel |
| 🧮 **Calculateur** | Évaluation d'un prix comptoir : prime %, spread %, verdict qualité |
| 📊 **Macro & Technique** | Score macro global, DXY, taux réels, RSI, Bollinger, saisonnalité |
| 🤖 **Prédictions ML** | Entraînement RF/XGB/LSTM/Stacking, signal à 5j/15j/30j, conviction |
| 💰 **Simulateur** | Simulation d'investissement historique + projection Monte-Carlo |
| 📈 **Backtesting P&L** | Equity curve, Sharpe, drawdown, alpha vs Buy & Hold |
| 📐 **Benchmark ML** | Comparaison multi-métriques, radar chart, test Diebold-Mariano |
| ⚙️ **Config** | Paramétrage persistant (SQLite) |

---

## 🔬 Méthodologie ML

### Walk-Forward Cross-Validation
Contrairement à un simple `train_test_split`, la **walk-forward** divise les données en fenêtres temporelles successives — **zéro look-ahead bias**.

```
Fold 1 : train [2019-2021] → test [2022]
Fold 2 : train [2019-2022] → test [2023]
Fold 3 : train [2019-2023] → test [2024]
```

### 6 modèles en concurrence

| Modèle | Type | Force |
|--------|------|-------|
| Random Walk | Naïf | Benchmark de référence |
| ARIMA | Statistique | Séries temporelles classiques |
| Random Forest | Ensemble | Robuste, interprétable |
| XGBoost | Boosting | Performant, rapide |
| LSTM bidirectionnel | Deep Learning | Séquences longues |
| Stacking | Méta-apprenant | Combine les 4 modèles |

### 40+ Features engineered
- **Prix** : log-rendements, volatilité réalisée, SMA 20/50/200j
- **Momentum** : RSI, MACD, Williams %R, CCI
- **Macro** : DXY (dollar), taux réels US, ratio Or/Argent
- **Risque** : VIX, spread 10Y-2Y, momentum SP500
- **Saisonnalité** : mois, jour de semaine (encodé cyclique)

### Métriques d'évaluation
- **DA%** (Directional Accuracy) — sens prédit vs sens réalisé
- **Brier Score** — qualité des probabilités (0 = parfait)
- **Sharpe Ratio** — rendement ajusté au risque
- **Alpha** — surperformance vs Buy & Hold
- **Test Diebold-Mariano** — significativité statistique des différences de modèles

---

## 🛠️ Stack technique

```
Python 3.11+
Streamlit ≥ 1.32      Interface web multi-pages (st.navigation)
scikit-learn          Random Forest, walk-forward CV
XGBoost               Gradient boosting
PyTorch (CPU)         LSTM bidirectionnel
statsmodels           ARIMA, tests ADF/Diebold-Mariano
yfinance + FRED API   Données marché temps réel
Plotly                Visualisations interactives
SQLite                Config persistante
python-dotenv         Gestion des secrets
openpyxl              Export Excel
```

**Bonus :** l'application est configurée comme une **PWA** (Progressive Web App) avec `manifest.json` + Service Worker — installable sur mobile.

---

## ⚙️ Installation locale

### Prérequis
- Python 3.11+
- Clé API FRED (gratuite sur [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))

### Setup

```bash
# 1. Cloner le repo
git clone https://github.com/BadreddineEK/goldSignal.git
cd goldSignal

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les secrets
cp .env.example .env
# Editer .env avec votre clé FRED_API_KEY

# 5. Lancer l'application
streamlit run app.py
```

### Variables d'environnement

```env
FRED_API_KEY=your_fred_api_key_here
```

---

## 📁 Structure du projet

```
goldSignal/
├── app.py                    # Point d'entrée — init DB, navigation Streamlit
├── requirements.txt
├── manifest.json             # PWA manifest
├── sw.js                     # Service Worker
├── pages/
│   ├── 00_accueil.py         # Dashboard principal + spots temps réel
│   ├── 01_calculateur.py     # Évaluation prix comptoir
│   ├── 02_macro.py           # Analyse macro & technique
│   ├── 03_predictions.py     # Modèles ML + signaux
│   ├── 04_portfolio.py       # Simulateur d'investissement
│   ├── 05_config.py          # Configuration
│   ├── 06_benchmark.py       # Benchmark multi-modèles
│   └── 07_backtest.py        # Backtesting P&L
├── data/
│   ├── fetcher.py            # Récupération données yfinance/FRED
│   └── database.py           # Init & seed SQLite
├── models/                   # Modèles ML sérialisés
├── analysis/                 # Modules d'analyse
├── utils/
│   ├── alerts.py             # Système d'alertes
│   ├── export.py             # Export Excel/CSV
│   └── formatting.py         # Formatage monétaire
├── config/
│   └── default_config.json   # Configuration par défaut
└── .streamlit/
    └── config.toml           # Thème Streamlit
```

---

## 🔗 Liens

- 🌐 **Application live** : [goldsignaltime.streamlit.app](https://goldsignaltime.streamlit.app/)
- 👤 **Portfolio** : [BadreddineEK — GitHub](https://github.com/BadreddineEK)
- 💼 **LinkedIn** : [badreddine-el-khamlichi](https://www.linkedin.com/in/badreddine-el-khamlichi/)

---

## 👤 Auteur

**Badreddine EL KHAMLICHI**  


---

*⚠️ Avertissement : GoldSignal est un outil pédagogique. Il ne constitue pas un conseil en investissement. Les performances passées ne préjugent pas des performances futures.*

# deep_4_all

Cours et codes pour enseigner le deep learning.

## Prérequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets Python)

## Installation

### 1. Installer uv

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Cloner le projet

```bash
git clone https://github.com/blancsw/deep_4_all.git
cd deep_4_all
```

### 3. Installer les dépendances

pour mac ou si vous avez que un processeur Intel pas de GPU 

```bash
uv sync
```

Si vous avre un GPU nvidia

Cette commande crée automatiquement un environnement virtuel et installe toutes les dépendances.

Pour installer pytorch GPU

````bash
uv sync --extra cu130
````

## Utilisation pour les Cours

```bash
uv run marimo edit
```

Activer l'environnement et lancer Jupyter Lab :

```bash
uv run jupyter lab
```

## Structure du dossier `cours/`

```
cours/
├── CM/                          # Cours Magistraux (notebooks Marimo)
│   ├── 01_cours_neural_networks.py   # Introduction aux réseaux de neurones
│   ├── 02_word_embedding.py          # Word embeddings
│   ├── 03_LSTM_RNN.py                # LSTM et réseaux récurrents
│   └── 04_transformers.py            # Transformers et self-attention
│
└── TP/                          # Travaux Pratiques
    ├── tp1_micrograd/           # Autograd from scratch (micrograd)
    ├── tp2/                     # TP2-3 : PyTorch MLP et généralisation
    └── tp4/                     # Distillation de modèles (DASD)
```

### CM - Cours Magistraux

Les cours sont des notebooks [Marimo](https://marimo.io/) interactifs.

| CM | Sujet | Description |
|----|-------|-------------|
| **CM1** | Réseaux de neurones | Perceptron, MLP, fonctions d'activation, backpropagation |
| **CM2** | Word Embeddings | Représentations vectorielles, Word2Vec, similarité sémantique |
| **CM3** | LSTM & RNN | Réseaux récurrents, LSTM, GRU, séquences |
| **CM4** | Transformers | Self-attention, architecture Transformer, positional encoding |

Pour lancer un cours :

```bash
uv run marimo run cours/CM/01_cours_neural_networks.py
```

### TP - Travaux Pratiques

| TP | Sujet | Description |
|----|-------|-------------|
| **TP1** | Micrograd | Implémentation de l'autograd from scratch, introduction à PyTorch |
| **TP2-3** | PyTorch MLP | Entraînement de MLP, optimisation, régularisation, leaderboard |
| **TP4** | DASD | Distillation de modèles de raisonnement (Long-CoT) |

> **Note :** Le TP3 (LSTM, embeddings, RNN) est intégré au cours `03_LSTM_RNN.py`.

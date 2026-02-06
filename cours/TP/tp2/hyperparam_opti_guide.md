# Hyperparameter Optimization Guide

## Overview

Instead of manually tweaking hyperparameters in `run_model.py`, you can now use automated optimization scripts to find the best configuration for your model.

Two scripts are provided:
 - **`optimize_hyperparams.py`** - Advanced optimization using Optuna (requires installation)

Both scripts focus on finding **small models** that generalize well, as mentioned in the challenge guidelines.

## Quick Start

First, install Optuna:
```bash
uv pip install optuna optuna-dashboard
```

Then run optimization:
```bash
# Run 50 trials of optimization
uv run optimize_hyperparams.py --n_trials 50

# Continue existing study
uv run optimize_hyperparams.py --n_trials 30 --load_if_exists

# View results in dashboard (optional)
optuna-dashboard sqlite:///oracle_optimization.db
```

## How It Works

### Search Space

The optimization explores these hyperparameter ranges:

| Parameter | Range | Notes |
|-----------|-------|-------|
| `hidden_dim` | 4, 8, 16, 32, 64 | Small models preferred |
| `num_layers` | 1-4 | Shallow networks to avoid overfitting |
| `dropout` | 0.0-0.6 | Regularization |
| `batch_size` | 32, 64, 128, 256 | |
| `epochs` | 100-300 | |
| `learning_rate` | 1e-4 to 1e-1 | Log scale |
| `weight_decay` | 0.0-0.5 | L2 regularization |
| `lambda_l1` | 0.0-0.01 | L1 regularization |
| `lambda_l2` | 0.0-0.05 | Additional L2 |

### Scoring Function

The optimization maximizes:
```
score = validation_accuracy - overfitting_penalty
```

Where `overfitting_penalty` penalizes models with large train-val gaps (>15%), encouraging generalization.

## Understanding Results

Optuna provides additional insights:
- **Optimization history plot** - Shows improvement over trials
- **Parameter importance** - Which parameters matter most
- **Parallel coordinate plot** - Visualize relationships

Results saved to `optimization_results/`:
- `oracle_optimization_best_params.json` - Best configuration
- `oracle_optimization_history.png` - Optimization plot
- `oracle_optimization_importance.png` - Feature importance
- `oracle_optimization.db` - SQLite database with all trials

## Key Insights from the Challenge

### Why Small Models?

The challenge includes a "twist": test data comes from **Terres Maudites** (Cursed Lands) where rules differ from training data. Large models that memorize training patterns will fail!

**Signs of good generalization:**
- Small train-val gap (<15%)
- Small model size (few parameters)
- Good validation accuracy

**Signs of overfitting:**
- Large train-val gap (>20%)
- Perfect training accuracy but poor validation
- Large model (hidden_dim=256, many layers)

### The Normalization Trick

The challenge hints: *"Dans la terre maudites l'atmosphère normalise les données de test"*

This means test data is normalized! Always train with `--normalize` flag.

## Example Workflow

```bash
# 1. Quick exploration (5 minutes)
uv run optimize_simple.py --mode random --n_trials 10

# 2. Review results, adjust search space if needed

# 3. Deeper search (30 minutes)
uv run optimize_simple.py --mode random --n_trials 50

# 4. Optional: Install Optuna for advanced optimization
uv pip install optuna

# 5. Fine-tune with intelligent search (1 hour)
uv run optimize_hyperparams.py --n_trials 100

# 6. Train final model with best params
uv run train_oracle.py --normalize --shuffle --optimizer adam --scheduler cosine \
  --hidden_dim 16 --num_layers 2 --dropout 0.3 \
  --batch_size 128 --epochs 200 --learning_rate 0.01 \
  --weight_decay 0.1 --lambda_l1 0.003 --lambda_l2 0.02

# 7. Submit checkpoints/best_model.pt to leaderboard
```

## Troubleshooting

### Error: "Training failed"
- Check that data files exist in `data/` directory
- Verify you're in the correct directory
- Try with default parameters first

### Search takes too long
- Reduce `--n_trials`
- Use `--mode random` instead of `grid`
- Reduce `epochs` range in search space

### All trials fail
- Check if `train_oracle.py` works manually
- Verify data files are present
- Check error messages in output

## Advanced: Customizing Search Space

Edit the search space in the optimization scripts:

### In `optimize_hyperparams.py`:
```python
def objective(trial: Trial):
    # Modify suggest_* calls
    hidden_dim = trial.suggest_categorical('hidden_dim', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    ...
```

## Next Steps

After finding good hyperparameters:
1. Train multiple models with best config (check consistency)
2. Analyze what makes them work (small size? high dropout?)
3. Experiment with slight variations
4. Submit best model to the leaderboard!

Remember: The goal is **generalization**, not perfect training accuracy!

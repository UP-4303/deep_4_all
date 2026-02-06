import argparse
import json
import math
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import Trial

from run_dungeon_model import run_dungeon_model


def objective(trial: Trial) -> float:
    mode = 'lstm'
    embed_dim = trial.suggest_int('embed_dim', 4, 32, step=2)
    hidden_dim = trial.suggest_int('hidden_dim', 1, 8, step=1)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    bidirectional = True
    batch_size = 32
    epochs = 50
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = 'adam'
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.05)
    
    try:
        results = run_dungeon_model(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            mode=mode,
            bidirectional=bidirectional,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer=optimizer_name,
            weight_decay=weight_decay,
            early_stopping=True,
            patience=10,
            verbose=False,
            plot=True
        )
        
        val_acc = results['best_val_acc']
        gap = results['gap_train_val']
        train_acc = results['final_train_acc']
        num_params = results['num_params']
        
        val_accs = results['history']['val_acc']
        
        if val_acc < 0.55:
            raise optuna.TrialPruned(f"Failed to learn (val_acc={val_acc:.2f})")
        
        val_std = np.std(val_accs)
        val_variance_coef = val_std / (val_acc + 1e-8)
        late_cutoff = int(len(val_accs) * 0.8)
        late_val_std = np.std(val_accs[late_cutoff:]) if late_cutoff < len(val_accs) else val_std
        trial.set_user_attr('gap_train_val', gap)
        trial.set_user_attr('final_train_acc', train_acc)
        trial.set_user_attr('best_val_acc', val_acc)
        trial.set_user_attr('val_std', val_std)
        trial.set_user_attr('val_variance_coef', val_variance_coef)
        trial.set_user_attr('late_val_std', late_val_std)
        trial.set_user_attr('num_params', num_params)
        trial.set_user_attr('epochs_trained', results['epochs_trained'])
        
        overfitting_penalty = max(0, abs(gap) - 0.10) * 1.0
        stability_penalty = val_variance_coef * 0.5 + late_val_std * 0.3
        convergence_penalty = (0.75 - train_acc) * 0.5 if train_acc < 0.75 else 0.0
        complexity_penalty = (num_params / 50000) * 0.15
        score = val_acc - overfitting_penalty - stability_penalty - convergence_penalty - complexity_penalty
        trial.set_user_attr('score_breakdown', {
            'base_val_acc': val_acc,
            'overfitting_penalty': overfitting_penalty,
            'stability_penalty': stability_penalty,
            'convergence_penalty': convergence_penalty,
            'complexity_penalty': complexity_penalty,
            'final_score': score
        })
        
        return score
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"\nTrial failed: {e}")
        raise optuna.TrialPruned(f"Training error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Optimize dungeon model hyperparameters')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials (default: 50)')
    parser.add_argument('--study_name', type=str, default='dungeon_optimization',
                        help='Study name (default: dungeon_optimization)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds')
    parser.add_argument('--sampler', type=str, default='tpe',
                        choices=['tpe', 'random'],
                        help='Sampling strategy (default: tpe)')
    parser.add_argument('--load_if_exists', action='store_true',
                        help='Load existing study')
    
    args = parser.parse_args()
    
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15, multivariate=True)
    else:
        sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        load_if_exists=args.load_if_exists,
        storage=f'sqlite:///{args.study_name}.db'
    )
    
    print(f"\nOptimizing {args.study_name}: {args.n_trials} trials, {args.sampler} sampler")
    if args.load_if_exists and len(study.trials) > 0:
        print(f"Continuing from {len(study.trials)} existing trials (best: {study.best_value:.4f})")
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(
                    f"\nTrial {trial.number}: score={trial.value:.4f}, "
                    f"val_acc={trial.user_attrs.get('best_val_acc', 'N/A'):.4f}, "
                    f"params={trial.user_attrs.get('num_params', 'N/A'):,}"
                ) if trial.value is not None else print(f"Trial {trial.number} pruned")
            ]
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    print("\nOptimization complete.")
    best_trial = study.best_trial
    print(f"\nBest trial #{best_trial.number}: score={best_trial.value:.4f}")
    print(f"  val_acc={best_trial.user_attrs.get('best_val_acc', 'N/A'):.4f}")
    print(f"  gap={best_trial.user_attrs.get('gap_train_val', 'N/A'):.4f}")
    print(f"  params={best_trial.user_attrs.get('num_params', 'N/A'):,}")
    print(f"  stability={best_trial.user_attrs.get('val_variance_coef', 'N/A'):.4f}")
    print("\nBest params:")
    for key, value in sorted(best_trial.params.items()):
        print(f"  {key}: {value}")
    results_dir = Path(__file__).parent / "optimization_results"
    results_dir.mkdir(exist_ok=True)
    
    best_params_path = results_dir / f"{args.study_name}_best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'user_attrs': best_trial.user_attrs
        }, f, indent=2)
    
    print(f"\nSaved to: {best_params_path}")
    
    print("\nTop 5 trials:")
    
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_trials = trials_df.nlargest(5, 'value')
        cols = ['number', 'value', 'params_embed_dim', 
                'params_hidden_dim', 'params_dropout', 'params_bidirectional']
        available_cols = [c for c in cols if c in top_trials.columns]
        print(top_trials[available_cols].to_string(index=False))
    try:
        import matplotlib.pyplot as plt
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig1.savefig(results_dir / f"{args.study_name}_history.png", dpi=150, bbox_inches='tight')
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        fig2.savefig(results_dir / f"{args.study_name}_importance.png", dpi=150, bbox_inches='tight')
        fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig3.savefig(results_dir / f"{args.study_name}_parallel.png", dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to {results_dir}")
    except ImportError:
        print("\nInstall optuna[visualization] for plots")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
    
    print("\nRetrain command:")
    print("uv run train_dungeon_logs.py \\")
    for key, value in sorted(best_trial.params.items()):
        print(f"  --{key} {value} \\")
    print()


if __name__ == "__main__":
    main()

"""
Hyperparameter Optimization Script using Optuna

This script automatically searches for the best hyperparameters for the GuildOracle model.
It focuses on finding small models that generalize well to avoid overfitting.
"""

import argparse
import json
import sys
import math
from pathlib import Path

import numpy as np

import optuna
from optuna.trial import Trial

from run_model import run_model


def objective(trial: Trial) -> float:
    """
    Objective function for Optuna optimization.
    
    Returns the validation accuracy (to maximize).
    We also penalize models that overfit (large gap between train and val).
    """
    
    # Sample hyperparameters
    # REFINED SEARCH SPACE based on importance analysis:
    # - Focus on hidden_dim (0.27) and weight_decay (0.29) - most important
    # - Narrow less important parameters
    
    # Most important: hidden_dim - expand range around best (7)
    # hidden_dim = trial.suggest_int('hidden_dim', 4, 12)  # Focused around 7
    
    # num_layers - best was 1, keep minimal range
    # num_layers = trial.suggest_int('num_layers', 2, 4)  # Reduced from 2-4
    
    # dropout - moderate importance (0.12)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)  # Reduced upper bound
    
    # batch_size - low importance (0.05), fix to best value
    # batch_size = trial.suggest_categorical('batch_size', [128, 256])  # Reduced choices
    
    # epochs - low importance (0.08), narrow range
    # epochs = trial.suggest_int('epochs', 100, 150)  # Reduced from 100-200
    
    # learning_rate - very low importance (0.02), fix range
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)  # Narrower range
    
    # Most important: weight_decay (0.29) - keep wide range
    weight_decay = trial.suggest_float('weight_decay', 0.05, 0.5)
    
    # lambda_l1 - moderate importance (0.11)
    lambda_l1 = trial.suggest_float('lambda_l1', 0.001, 0.02)
    
    # lambda_l2 - low importance (0.06)
    lambda_l2 = trial.suggest_float('lambda_l2', 0.001, 0.05)
    
    try:
        # Run training
        results = run_model(
            hidden_dim=4,
            num_layers=2,
            dropout=dropout,
            batch_size=128,
            epochs=100,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
        )
        
        val_acc = results['best_val_acc']
        gap = results['gap_train_val']
        train_acc = results['history']['train_acc'][-1]

        val_accs = results['history']['val_acc']
        train_accs = results['history']['train_acc']
        
        # Early failure detection - prune trials that fail to learn
        if val_acc < 0.8:  # Less than 80% = probably failed
            raise optuna.TrialPruned(f"Trial failed to learn (val_acc={val_acc:.2f})")
        
        # Calculate stability metrics
        val_std = np.std(val_accs)
        val_variance_coef = val_std / (val_acc + 1e-8)  # Coefficient of variation
        
        # Calculate late-stage stability (last 20% of epochs)
        late_cutoff = int(len(val_accs) * 0.8)
        late_val_std = np.std(val_accs[late_cutoff:]) if late_cutoff < len(val_accs) else val_std
        
        # Report comprehensive results
        trial.set_user_attr('gap_train_val', gap)
        trial.set_user_attr('final_train_acc', train_acc)
        trial.set_user_attr('best_val_acc', val_acc)
        trial.set_user_attr('val_std', val_std)
        trial.set_user_attr('val_variance_coef', val_variance_coef)
        trial.set_user_attr('late_val_std', late_val_std)
        
        # IMPROVED SCORING FUNCTION
        # 1. Overfitting penalty (gap > 15%)
        overfitting_penalty = max(0, math.fabs(gap) - 0.15) * 0.8  # Increased weight
        
        # 2. Instability penalty (shaky curves)
        # Penalize both overall variance and late-stage instability
        stability_penalty = val_variance_coef * 0.5 + late_val_std * 0.3
        
        # 3. Convergence penalty (didn't train enough)
        # If train acc is low, model didn't converge
        if train_acc < 0.85:
            convergence_penalty = (0.85 - train_acc) * 0.3
        else:
            convergence_penalty = 0.0
        
        # Final score
        score = val_acc - overfitting_penalty - stability_penalty - convergence_penalty
        
        # Log detailed score breakdown
        trial.set_user_attr('score_breakdown', {
            'base_val_acc': val_acc,
            'overfitting_penalty': overfitting_penalty,
            'stability_penalty': stability_penalty,
            'convergence_penalty': convergence_penalty,
            'final_score': score
        })
        
        return score
        
    except optuna.TrialPruned:
        # Re-raise pruned trials (this is expected)
        raise
    except Exception as e:
        print(f"\n❌ Trial failed with error: {e}")
        # Prune completely failed trials instead of returning 0
        raise optuna.TrialPruned(f"Training error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    parser.add_argument('--study_name', type=str, default='oracle_optimization',
                        help='Name of the Optuna study')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for the entire optimization')
    parser.add_argument('--sampler', type=str, default='tpe',
                        choices=['tpe', 'random', 'grid'],
                        help='Sampling strategy (default: tpe)')
    parser.add_argument('--load_if_exists', action='store_true',
                        help='Load existing study if it exists')
    
    args = parser.parse_args()
    
    # Create sampler with improved settings
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=10,  # Random exploration first
            multivariate=True,    # Consider parameter interactions
        )
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        # For grid sampler, we'd need to define search space
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # We want to maximize validation accuracy
        sampler=sampler,
        load_if_exists=args.load_if_exists,
        storage=f'sqlite:///{args.study_name}.db'  # Persist study to database
    )
    
    print("=" * 80)
    print(f"Starting Hyperparameter Optimization: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Sampler: {args.sampler}")
    print(f"Existing trials: {len(study.trials)}")
    if args.load_if_exists and len(study.trials) > 0:
        print(f"Continuing from existing study with {len(study.trials)} trials")
        print(f"Current best score: {study.best_value:.4f}")
    print("\n🎯 REFINED SEARCH STRATEGY:")
    print("  - Focus on hidden_dim (4-12) and weight_decay (0-0.5)")
    print("  - Narrow less important parameters")
    print("  - Early pruning of failed trials (<60% acc)")
    print("  - Enhanced stability and convergence penalties")
    print("=" * 80)
    print()
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(
                    f"\n{'='*60}\n"
                    f"Trial {trial.number} finished: Score = {trial.value:.4f}\n"
                    f"  Val Acc: {trial.user_attrs.get('best_val_acc', 'N/A')}\n"
                    f"  Gap: {trial.user_attrs.get('gap_train_val', 'N/A')}\n"
                    f"  Stability: {trial.user_attrs.get('late_val_std', 'N/A')}\n"
                    f"{'='*60}"
                ) if trial.value is not None else print(f"\nTrial {trial.number} pruned")
            ]
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"  Best Score: {best_trial.value:.4f}")
    
    if 'best_val_acc' in best_trial.user_attrs:
        print(f"  Actual Val Accuracy: {best_trial.user_attrs['best_val_acc']:.4f}")
    if 'gap_train_val' in best_trial.user_attrs:
        print(f"  Train-Val Gap: {best_trial.user_attrs['gap_train_val']:.4f}")
    if 'val_variance_coef' in best_trial.user_attrs:
        print(f"  Stability (variance coef): {best_trial.user_attrs['val_variance_coef']:.4f}")
    if 'late_val_std' in best_trial.user_attrs:
        print(f"  Late-stage stability (std): {best_trial.user_attrs['late_val_std']:.4f}")
    
    print("\n  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    results_dir = Path(__file__).parent / "optimization_results"
    results_dir.mkdir(exist_ok=True)
    
    best_params_path = results_dir / f"{args.study_name}_best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'user_attrs': best_trial.user_attrs
        }, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Print top 5 trials
    print("\n" + "-" * 80)
    print("Top 5 Trials:")
    print("-" * 80)
    
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_trials = trials_df.nlargest(5, 'value')
        print(top_trials[['number', 'value', 'params_dropout', 'params_learning_rate', 'params_weight_decay', 'params_lambda_l1', 'params_lambda_l2']].to_string(index=False))
    
    # Generate optimization visualization
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig1.savefig(results_dir / f"{args.study_name}_history.png", dpi=150, bbox_inches='tight')
        print(f"\nOptimization history plot saved to: {results_dir / f'{args.study_name}_history.png'}")
        
        # Plot parameter importances
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        fig2.savefig(results_dir / f"{args.study_name}_importance.png", dpi=150, bbox_inches='tight')
        print(f"Parameter importance plot saved to: {results_dir / f'{args.study_name}_importance.png'}")
        
        # Plot parallel coordinate
        fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig3.savefig(results_dir / f"{args.study_name}_parallel.png", dpi=150, bbox_inches='tight')
        print(f"Parallel coordinate plot saved to: {results_dir / f'{args.study_name}_parallel.png'}")
        
    except ImportError:
        print("\nNote: Install optuna[visualization] and matplotlib for plots")
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")
    
    print("\n" + "=" * 80)
    print("To retrain with best parameters:")
    print("=" * 80)
    print(f"\nuv run train_oracle.py --normalize --shuffle --optimizer adam --scheduler cosine \\")
    for key, value in best_trial.params.items():
        print(f"  --{key} {value} \\")
    print()


if __name__ == "__main__":
    main()

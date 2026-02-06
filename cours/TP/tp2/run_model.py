import os
from pathlib import Path
import json

checkpoint_dir = Path(__file__).parent / "checkpoints"

def run_model(
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        lambda_l1: float,
        lambda_l2: float,
) -> dict:
    """
    Docstring for run_model
    
    returns: {
        'best_val_acc': float,
        'parameters': dict,
        'history': {
            'train_acc': list[float],
            'val_acc': list[float],
            'train_loss': list[float],
            'val_loss': list[float]
        },
        'gap_train_val': float
    }
    """
    res = os.system(f"uv run train_oracle.py --normalize --shuffle --optimizer adam --scheduler cosine --hidden_dim {hidden_dim} --num_layers {num_layers} --dropout {dropout} --batch_size {batch_size} --epochs {epochs} --learning_rate {learning_rate} --weight_decay {weight_decay} --lambda_l1 {lambda_l1} --lambda_l2 {lambda_l2}")

    if res != 0:
        raise RuntimeError("Training failed")
    
    # Lecture des résultats
    with open(checkpoint_dir / "res.json", 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    hidden_dim = 11
    num_layers = 2
    dropout = 0.4
    batch_size = 128
    epochs = 200
    learning_rate = 0.0005
    weight_decay = 0.4
    lambda_l1 = 0.002
    lambda_l2 = 0.025

    data = run_model(
        hidden_dim,
        num_layers,
        dropout,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        lambda_l1,
        lambda_l2,
    )
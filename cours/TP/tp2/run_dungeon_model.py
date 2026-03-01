import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

from train_dungeon_logs import DungeonLogDataset, train_epoch, evaluate, evaluate_by_category, plot_history
from baseline_model import DungeonOracle, count_parameters


def run_dungeon_model(
    embed_dim: int = 258,
    hidden_dim: int = 258,
    num_layers: int = 1,
    dropout: float = 0.0,
    mode: str = 'linear',
    bidirectional: bool = False,
    num_heads: int = 2,
    batch_size: int = 32,
    epochs: int = 6,
    learning_rate: float = 0.1,
    optimizer: str = 'sgd',
    weight_decay: float = 0.0,
    use_scheduler: bool = False,
    early_stopping: bool = False,
    patience: int = 10,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Device: {device}")
    
    data_dir = Path(__file__).parent / "data"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    vocab_path = data_dir / "vocabulary_dungeon.json"
    train_path = data_dir / "train_dungeon.csv"
    val_path = data_dir / "val_dungeon.csv"
    
    val_df = pd.read_csv(val_path)
    
    if verbose:
        print("\nChargement des données...")
    
    train_dataset = DungeonLogDataset(str(train_path), str(vocab_path))
    val_dataset = DungeonLogDataset(str(val_path), str(vocab_path))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    if verbose:
        print(f"Train: {len(train_dataset)} séquences")
        print(f"Val: {len(val_dataset)} séquences")
        print(f"Vocabulaire: {train_dataset.vocab_size} tokens")
        
        train_lengths = train_dataset.lengths.numpy()
        print(f"Longueur des séquences: min={train_lengths.min()}, "
              f"max={train_lengths.max()}, mean={train_lengths.mean():.1f}")
    
    if verbose:
        print("\nCréation du modèle...")
    
    model = DungeonOracle(
        vocab_size=train_dataset.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        mode=mode,
        max_length=train_dataset.max_length,
        bidirectional=bidirectional,
        padding_idx=train_dataset.pad_idx,
        num_heads=num_heads,
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    
    if verbose:
        print(f"Architecture: {mode}")
        print(f"Bidirectionnel: {bidirectional}")
        print(f"Paramètres: {num_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    
    if optimizer == 'adam':
        opt = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        opt = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    if verbose:
        print(f"Optimiseur: {optimizer.upper()}, LR: {learning_rate}")
    
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=5
        )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Début de l'entraînement")
        print("=" * 60)
    
    best_val_acc = 0
    patience_counter = 0
    
    epoch_iterator = tqdm.trange(epochs) if verbose else range(epochs)
    
    for epoch in epoch_iterator:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, opt, device
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, checkpoint_dir / "best_dungeon_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if early_stopping and patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping après {epoch + 1} epochs")
            break
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Meilleure accuracy validation: {best_val_acc:.2%}")
        print(f"Modèle sauvegardé: {checkpoint_dir / 'best_dungeon_model.pt'}")
        print("=" * 60)
        
        with open(checkpoint_dir / "dungeon_history.json", 'w') as f:
            json.dump(history, f, indent=4)
        
        print("\n" + "-" * 60)
        print("Analyse par catégorie de donjon")
        print("-" * 60)
        
        cat_results = evaluate_by_category(model, val_loader, device, val_df)
        
        for cat, stats in sorted(cat_results.items()):
            print(f"  {cat:30s}: {stats['accuracy']:.2%} ({stats['count']} ex.)")
        
        print("\n" + "!" * 60)
        print("POINTS D'ATTENTION:")
        
        gap = history['train_acc'][-1] - history['val_acc'][-1]
        if gap > 0.10:
            print(f"  - OVERFITTING: Gap train-val de {gap:.2%}")
            print("    -> Augmentez dropout, reduisez hidden_dim, ou ajoutez regularisation")
        
        print("!" * 60)
        
    if plot:
        plot_history(history, checkpoint_dir / "dungeon_training_curves.png")
    
    final_train_acc = history['train_acc'][-1]
    gap_train_val = final_train_acc - best_val_acc
    
    results = {
        'best_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'gap_train_val': gap_train_val,
        'history': history,
        'num_params': num_params,
        'epochs_trained': len(history['train_acc'])
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de l'Oracle du Dongeon")
    
    parser.add_argument('--embed_dim', type=int, default=258,
                        help='Dimension des embeddings')
    parser.add_argument('--hidden_dim', type=int, default=258,
                        help='Dimension de l\'état caché RNN/LSTM')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Nombre de couches RNN/LSTM')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout entre les couches RNN')
    parser.add_argument('--mode', type=str, default='linear',
                        choices=['linear', 'rnn', 'lstm', 'attention'],
                        help='Architecture du modèle')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='RNN/LSTM bidirectionnel')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Nombre de têtes d\'attention (pour mode attention)')
    
    parser.add_argument('--epochs', type=int, default=6,
                        help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd'],
                        help='Optimiseur')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                        help='Utiliser un learning rate scheduler')
    
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Activer early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience pour early stopping')
    
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Générer les courbes de training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ORACLE DU DONJON - Entraînement")
    print("=" * 60)
    print("\nConfiguration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)
    
    results = run_dungeon_model(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=args.mode,
        bidirectional=args.bidirectional,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        early_stopping=args.early_stopping,
        patience=args.patience,
        plot=args.plot,
        verbose=True,
    )

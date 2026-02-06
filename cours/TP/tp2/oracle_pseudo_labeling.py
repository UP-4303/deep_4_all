"""
Application Gradio : Tournoi Oracle - Leaderboard

Interface web pour le dataset Oracle (aventuriers avec features tabulaires).

Usage:
    python app_leaderboard_oracle.py
    # Ouvre http://localhost:7860
"""

from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_oracle import AdventurerDataset


# =============================================================================
# Évaluateur spécifique Oracle
# =============================================================================

class OraclePseudoLabeler():
    """Pseudo-labeler pour le dataset Oracle (features tabulaires)."""
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(checkpoint_path, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

    def generate_pseudo_labels(self, data_path: str, output_path: str, threshold: float = 0.1, noise_std: float = 0.1):
        """Génère des pseudo-labels pour une variation perturbée d'un dataset."""
        dataset = AdventurerDataset(data_path, normalize=True)
        noised_features = dataset.features + torch.randn_like(dataset.features) * noise_std
        noised_features = noised_features.to(self.device)
        with torch.no_grad():            
            logits = self.model(noised_features)
            probs = torch.sigmoid(logits).squeeze()

            # Filtrer les prédictions incertaines
            confident_mask = (probs < threshold) | (probs > 1 - threshold)
            pseudo_labels = (probs > 0.5).int()
            
            # Garder seulement les données et labels confiants
            confident_data = noised_features[confident_mask]
            confident_labels = pseudo_labels[confident_mask]

        # Dénormaliser les données si elles ont été normalisées
        if hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
            confident_data = confident_data * torch.tensor(dataset.std, dtype=torch.float32, device=self.device) + torch.tensor(dataset.mean, dtype=torch.float32, device=self.device)
        
        # Créer un DataFrame avec les données perturbées et les pseudo-labels
        column_names = ['force', 'intelligence', 'agilite', 'chance', 'experience', 'niveau_quete', 'equipement', 'fatigue']
        df = pd.DataFrame(confident_data.cpu().numpy(), columns=column_names)
        df['survie'] = confident_labels.cpu().numpy()

        # Sauvegarder le CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        print(f"Dataset perturbé avec {len(df)} exemples confiants sauvegardé dans {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Pseudo-labeler pour le dataset Oracle"
            )
    parser.add_argument(
            '--checkpoint_path', type=str, required=True,
            help='Chemin vers le checkpoint du modèle entraîné'
            )
    parser.add_argument(
            '--data_path', type=str, required=True,
            help='Chemin vers le dataset tabulaire à pseudo-labéliser (CSV)'
            )
    parser.add_argument(
            '--output_path', type=str, required=True,
            help='Chemin de sortie pour le dataset pseudo-labélisé (CSV)'
            )
    parser.add_argument(
            '--threshold', type=float, default=0.1,
            help='Seuil de confiance pour les pseudo-labels'
            )
    parser.add_argument(
            '--noise_std', type=float, default=0.1,
            help='Écart-type du bruit ajouté aux features'
            )
    args = parser.parse_args()

    evaluator = OraclePseudoLabeler(args.checkpoint_path)
    evaluator.generate_pseudo_labels(
        data_path=args.data_path,
        output_path=args.output_path,
        threshold=args.threshold,
        noise_std=args.noise_std
    )

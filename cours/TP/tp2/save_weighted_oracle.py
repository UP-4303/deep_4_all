"""
Script pour créer un modèle Oracle avec une formule pondérée manuelle.

Ce modèle utilise des poids prédéfinis au lieu d'être entraîné.
"""

import torch
from baseline_model import GuildOracle
from pathlib import Path


def create_weighted_oracle():
    """
    Crée un GuildOracle avec des poids manuels basés sur une formule pondérée.
    
    Poids appliqués:
    - Intelligence: 30%
    - Agilité: 20%
    - Chance: 20%
    - Équipement: 15%
    - Force: 10%
    - Expérience: 5%
    - Fatigue: -10%
    - Difficulté (niveau_quete): -10%
    - Arrogance (force > 70): -15% (approximé via architecture non-linéaire)
    """
    
    # Créer un modèle avec architecture permettant la non-linéarité
    # On utilise 2 couches pour pouvoir approximer la condition d'arrogance
    model = GuildOracle(
        input_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0
    )
    
    model.eval()
    
    # Order des features dans train.csv:
    # 0: force, 1: intelligence, 2: agilite, 3: chance, 
    # 4: experience, 5: niveau_quete, 6: equipement, 7: fatigue
    
    with torch.no_grad():
        # Première couche: extraction des features
        # On crée des neurons pour détecter force > 70 et les autres stats
        first_layer = model.network[0]  # Linear(8, 16)
        
        # Initialiser à zéro
        first_layer.weight.zero_()
        first_layer.bias.zero_()
        
        # Neurons 0-7: copient les features de base avec leurs poids
        first_layer.weight[0, 0] = 0.10   # force
        first_layer.weight[1, 1] = 0.30   # intelligence
        first_layer.weight[2, 2] = 0.20   # agilite
        first_layer.weight[3, 3] = 0.20   # chance
        first_layer.weight[4, 4] = 0.05   # experience
        first_layer.weight[5, 5] = -0.10  # niveau_quete (difficulté)
        first_layer.weight[6, 6] = 0.15   # equipement
        first_layer.weight[7, 7] = -0.10  # fatigue
        
        # Neurons 8-9: détecteurs pour arrogance (force > 70)
        # Neuron 8: détecte quand force > 70
        first_layer.weight[8, 0] = 1.0    # copie force
        first_layer.bias[8] = -70.0       # décalage pour activation si force > 70
        
        # Neuron 9: neuron négatif pour l'arrogance
        first_layer.weight[9, 0] = 1.0
        first_layer.bias[9] = -70.0
        
        # Dernière couche: agrégation
        # On saute BatchNorm, ReLU, Dropout pour accéder au dernier Linear
        last_layer = model.network[-1]  # Linear(16, 1)
        last_layer.weight.zero_()
        last_layer.bias.zero_()
        
        # Sommer les 8 premières neurons (stats de base)
        for i in range(8):
            last_layer.weight[0, i] = 1.0
        
        # Appliquer pénalité d'arrogance via ReLU
        # Après ReLU, neuron 8 sera > 0 seulement si force > 70
        # On applique un poids négatif pour la pénalité
        last_layer.weight[0, 8] = -0.15  # pénalité d'arrogance
    
    return model


def main():
    """Crée et sauvegarde le modèle."""
    print("Création du modèle Oracle avec formule pondérée...")
    
    model = create_weighted_oracle()
    
    # Sauvegarder
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / "weighted_formula_oracle.pt"
    
    torch.save(model, save_path)
    
    print(f"\n✓ Modèle sauvegardé: {save_path}")
    print("\nPoids appliqués:")
    print("  • Intelligence: 30%")
    print("  • Agilité: 20%")
    print("  • Chance: 20%")
    print("  • Équipement: 15%")
    print("  • Force: 10%")
    print("  • Expérience: 5%")
    print("  • Fatigue: -10%")
    print("  • Niveau quête (difficulté): -10%")
    print("  • Arrogance (force > 70): -15%")
    
    # Test rapide
    print("\nTest du modèle:")
    test_cases = [
        {
            "name": "Aventurier équilibré (force=50)",
            "stats": torch.tensor([[50.0, 60.0, 50.0, 50.0, 10.0, 5.0, 40.0, 30.0]])
        },
        {
            "name": "Aventurier arrogant (force=80)",
            "stats": torch.tensor([[80.0, 60.0, 50.0, 50.0, 10.0, 5.0, 40.0, 30.0]])
        }
    ]
    
    model.eval()
    for test in test_cases:
        with torch.no_grad():
            logits = model(test["stats"])
            proba = torch.sigmoid(logits).item()
        print(f"  {test['name']}: {proba:.2%} de survie")


if __name__ == "__main__":
    main()

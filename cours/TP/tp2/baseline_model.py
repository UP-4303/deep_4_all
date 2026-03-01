"""
Modèle Baseline : Oracle de la Guilde (Non Optimal)

Ce modèle est VOLONTAIREMENT non optimal.
Les étudiants doivent identifier et corriger les problèmes !

Contient:
- GuildOracle : MLP pour prédiction de survie (stats → survie)
- DungeonOracle : LSTM pour prédiction de survie (séquence d'événements → survie)
"""

import torch
import torch.nn as nn


# ============================================================================
# TP2 : Modèle MLP pour stats d'aventuriers
# ============================================================================


class GuildOracle(nn.Module):
    """
    Modèle baseline pour prédire la survie des aventuriers.

    Architecture : MLP profond (trop profond !)
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, num_layers: int = 5, dropout: float = 0.5):
        """
        Args:
            input_dim: Nombre de features (8 stats)
            hidden_dim: Dimension des couches cachées
            num_layers: Nombre de couches cachées
            dropout: Taux de dropout pour régularisation
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers doit être au moins 1")

        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            # Couche d'entrée
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Dropout pour régularisation

            # Couches cachées
            for i in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())  # Alternance d'activations
                layers.append(nn.Dropout(dropout))  # Dropout pour régularisation

            # Couche de sortie
            layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, input_dim)

        Returns:
            Logits de shape (batch_size, 1)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x)
        return (proba > 0.5).float()

    def l1_regularization(self) -> torch.Tensor:
        """
        Calcule la pénalité L1 (somme des valeurs absolues des poids).
        
        À utiliser dans la fonction de perte:
            loss = criterion(outputs, targets) + lambda_l1 * model.l1_regularization()
        
        Returns:
            Pénalité L1 (scalaire)
        """
        l1_penalty = 0.0
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                l1_penalty += torch.sum(torch.abs(module.weight))
        return l1_penalty

    def l2_regularization(self) -> torch.Tensor:
        """
        Calcule la pénalité L2 (somme des carrés des poids).
        
        Note: PyTorch inclut L2 via weight_decay dans l'optimiseur.
        Cette méthode est utile pour un contrôle manuel.
        
        À utiliser dans la fonction de perte:
            loss = criterion(outputs, targets) + lambda_l2 * model.l2_regularization()
        
        Returns:
            Pénalité L2 (scalaire)
        """
        l2_penalty = 0.0
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                l2_penalty += torch.sum(module.weight ** 2)
        return l2_penalty


# ============================================================================
# TP3 : Modèle LSTM pour séquences de donjon
# ============================================================================


class PositionalEncoding(nn.Module):
    """
    Ajoute des informations de position aux embeddings.
    Nécessaire pour l'attention car elle n'a pas de notion intrinsèque de l'ordre.
    
    Utilise des fonctions sinus/cosinus pour encoder la position.
    """
    def __init__(self, embed_dim: int, max_length: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer une matrice de positions
        position = torch.arange(max_length).unsqueeze(1)  # (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe = torch.zeros(max_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Dimensions paires
        pe[:, 1::2] = torch.cos(position * div_term)  # Dimensions impaires
        
        # Shape: (1, max_length, embed_dim) pour broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (batch_size, seq_length, embed_dim)
        Returns:
            Tensor avec positions ajoutées, même shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DungeonOracle(nn.Module):
    """
    Modèle baseline pour prédire la survie à partir d'une séquence d'événements.

    Architectures disponibles:
    - linear: Embedding → Flatten → MLP
    - rnn: Embedding → RNN → Classifier  
    - lstm: Embedding → LSTM → Classifier
    - attention: Embedding → Positional Encoding → Self-Attention → Classifier (NOUVEAU)

    PROBLEMES VOLONTAIRES (à corriger par les étudiants):
    1. Embedding dimension trop petite (8) -> perd de l'information semantique
    2. Un seul layer LSTM -> difficile de capturer les patterns complexes
    3. Pas de Dropout -> risque d'overfitting
    4. Utilise RNN simple au lieu de LSTM -> vanishing gradient sur longues sequences
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 2,
            hidden_dim: int = 258,
            num_layers: int = 1,
            dropout: float = 0.0,
            mode: str = "linear",
            bidirectional: bool = False,
            padding_idx: int = 0,
            max_length: int = 140,
            num_heads: int = 2  # Nouveau paramètre pour attention
            ):
        """
        Args:
            vocab_size: Taille du vocabulaire (nombre d'événements uniques)
            embed_dim: Dimension des embeddings
            hidden_dim: Dimension de l'état caché du RNN/LSTM/Attention
            num_layers: Nombre de couches RNN/LSTM/Attention
            dropout: Dropout entre les couches (si num_layers > 1)
            mode: "linear", "rnn", "lstm", ou "attention"
            bidirectional: Si True, RNN bidirectionnel (non utilisé pour attention)
            padding_idx: Index du token de padding (ignoré dans les embeddings)
            max_length: Longueur maximale de séquence
            num_heads: Nombre de têtes d'attention (mode attention uniquement)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode.lower().strip()
        self.num_heads = num_heads
        self.max_length = max_length

        # Couche d'embedding : transforme les IDs en vecteurs denses
        # Le padding_idx=0 fait que le vecteur pour <PAD> reste à zéro
        self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx
                )

        # Approche Baseline Linéaire (Alternative au RNN)
        # On aplatit tout : (Batch, Seq_Len * Embed_Dim)
        self.solo_embeddings = nn.Sequential(
                nn.Flatten(),
                nn.Linear(max_length * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # Sortie directe pour comparaison
                )
        
        if self.mode == "attention":
            # Architecture Self-Attention (plus petite que LSTM)
            # AVANTAGES:
            # - Parallélisable (vs séquentiel pour RNN/LSTM)
            # - Capture les dépendances longues distance
            # - Moins de paramètres si bien configuré
            
            # Positional encoding (ajoute info de position)
            self.pos_encoder = PositionalEncoding(embed_dim, max_length, dropout)
            
            # Projection de embed_dim vers hidden_dim si nécessaire
            if embed_dim != hidden_dim:
                self.input_projection = nn.Linear(embed_dim, hidden_dim)
            else:
                self.input_projection = nn.Identity()
            
            # Multi-head self-attention
            # num_heads doit diviser hidden_dim
            assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) doit être divisible par num_heads ({num_heads})"
            
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
            # Layer normalization après chaque attention
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
            
            # Feed-forward network optionnel après attention
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )
        
        elif self.mode != "linear":
            # Couche récurrente (RNN/LSTM)
            # PROBLEME: Par défaut c'est un RNN simple qui souffre du vanishing gradient
            rnn_class = nn.LSTM if self.mode == "lstm" else nn.RNN

            self.rnn = rnn_class(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                    )

            # Couche de classification
            # Si bidirectionnel, on a 2x hidden_dim
            classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

            self.classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, 1)
                    )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, seq_length) contenant les IDs d'événements
            lengths: Tensor de shape (batch_size,) contenant les longueurs réelles
                     (optionnel, pour ignorer le padding)

        Returns:
            Logits de shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # Étape 1: Embedding
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        if self.mode == "linear":
            # Mode linéaire nécessite une longueur fixe
            # Pad ou truncate si nécessaire pour correspondre à max_length
            batch_size, seq_len, embed_dim = embedded.shape
            if seq_len != self.max_length:
                padded = torch.zeros(batch_size, self.max_length, embed_dim, 
                                    device=embedded.device, dtype=embedded.dtype)
                copy_len = min(seq_len, self.max_length)
                padded[:, :copy_len, :] = embedded[:, :copy_len, :]
                embedded = padded
            return self.solo_embeddings(embedded)
        
        elif self.mode == "attention":
            # Architecture Self-Attention
            
            # Étape 2: Positional encoding
            # (batch, seq_len, embed_dim) → (batch, seq_len, embed_dim)
            embedded = self.pos_encoder(embedded)
            
            # Étape 3: Projection vers hidden_dim si nécessaire
            # (batch, seq_len, embed_dim) → (batch, seq_len, hidden_dim)
            x_proj = self.input_projection(embedded)
            
            # Étape 4: Appliquer les couches d'attention avec connexions résiduelles
            for attention_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
                # Self-attention: query, key, value sont tous x_proj
                # attn_output: (batch, seq_len, hidden_dim)
                attn_output, _ = attention_layer(x_proj, x_proj, x_proj)
                
                # Connexion résiduelle + Layer Norm
                x_proj = layer_norm(x_proj + attn_output)
            
            # Étape 5: Feed-forward network avec résiduelle
            ffn_output = self.ffn(x_proj)
            x_proj = self.ffn_norm(x_proj + ffn_output)
            
            # Étape 6: Pooling - moyenne sur la séquence
            # (batch, seq_len, hidden_dim) → (batch, hidden_dim)
            # Alternative: prendre le premier token (comme BERT avec [CLS])
            pooled = x_proj.mean(dim=1)
            
            # Étape 7: Classification
            logits = self.classifier(pooled)
            
            return logits
        
        elif self.mode != "linear":
            # Étape 2: Passage dans le RNN/LSTM
            # output: (batch, seq_len, hidden_dim * num_directions)
            # hidden: (num_layers * num_directions, batch, hidden_dim)
            if self.mode == "lstm":
                output, (hidden, cell) = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded)

            # Étape 3: Extraire le dernier état caché
            # Pour un RNN standard, on prend la dernière sortie
            if self.bidirectional:
                # Concaténer forward et backward
                hidden_forward = hidden[-2]  # Dernière couche, direction forward
                hidden_backward = hidden[-1]  # Dernière couche, direction backward
                final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                # Juste la dernière couche
                final_hidden = hidden[-1]

            # Étape 4: Classification
            logits = self.classifier(final_hidden)

            return logits

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x, lengths)
        return (proba > 0.5).float()

    def get_embeddings(self) -> torch.Tensor:
        """Retourne les poids de la couche d'embedding pour visualisation."""
        return self.embedding.weight.detach().clone()


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Affiche un résumé du modèle."""
    print("=" * 50)
    print("Résumé du modèle")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    print("=" * 50)

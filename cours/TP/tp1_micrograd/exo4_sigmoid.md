# Exercice 4: Implementer Sigmoid, Log et Binary Cross-Entropy

> Master 2 Informatique - Introduction IA

## Objectif

Implementer la fonction d'activation sigmoid et la fonction log dans `engine.py`, puis utiliser ces fonctions pour creer une loss Binary Cross-Entropy (BCE) dans `exo3_mlp_training.py`.

---

## Partie 1: Fonction Sigmoid

### Formules

```
sigmoid(x) = 1 / (1 + e^(-x))

d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
```

### Proprietes

| Propriete | Valeur |
|-----------|--------|
| `sigmoid(0)` | 0.5 |
| `sigmoid(x)` quand x -> +infini | 1 |
| `sigmoid(x)` quand x -> -infini | 0 |
| Derivee maximale (en x=0) | 0.25 |

### Attention: Stabilite numerique

La formule naive `1 / (1 + exp(-x))` peut causer un **overflow** quand `x` est un grand nombre negatif (car `exp(-x)` devient tres grand).

**Solution**: Utiliser une version stable:
- Si `x >= 0`: utiliser `1 / (1 + exp(-x))`
- Si `x < 0`: utiliser `exp(x) / (1 + exp(x))` (mathematiquement equivalent)

### A faire dans `engine.py`

Completez la methode `sigmoid()` dans la classe `Value`.

---

## Partie 2: Fonction Log (Logarithme naturel)

### Formules

```
log(x) = ln(x)

d(log)/dx = 1/x
```

### Attention

- `log(0)` n'est pas defini -> ajouter un petit epsilon (ex: `1e-7`)
- La derivee `1/x` peut exploser si `x` est proche de 0

### A faire dans `engine.py`

Completez la methode `log()` dans la classe `Value`.

---

## Partie 3: Binary Cross-Entropy Loss

### Formule

Pour une classification binaire avec labels `y` dans {-1, +1}:

```
1. Convertir le label: t = (y + 1) / 2    # -1 -> 0, +1 -> 1
2. Calculer la probabilite: p = sigmoid(prediction)
3. BCE Loss: L = -t * log(p) - (1-t) * log(1-p)
```

### Interpretation

- Si `t = 1` (classe positive): on veut `p` proche de 1, donc `log(p)` proche de 0
- Si `t = 0` (classe negative): on veut `p` proche de 0, donc `log(1-p)` proche de 0

### A faire dans `exo3_mlp_training.py`

Completez la fonction `bce_loss(y, y_preds)` qui:
1. Parcourt les paires (label, prediction)
2. Convertit le label de -1/+1 vers 0/1
3. Applique sigmoid sur la prediction
4. Calcule la BCE pour chaque exemple
5. Retourne la liste des losses

---

## Verification

Lancez le script d'entrainement:
```bash
python exo3_mlp_training.py
```

Vous devriez observer:
- La loss diminue au fil des epochs
- La precision augmente vers ~100%
- Le graphique de decision separe bien les deux classes

---

## Bonus: Comparaison Hinge Loss vs BCE Loss

Modifiez `exo3_mlp_training.py` pour alterner entre `hinge_loss` et `bce_loss` et comparez:
- Vitesse de convergence
- Precision finale
- Forme de la frontiere de decision

### Differences theoriques

| Aspect | Hinge Loss (SVM) | BCE Loss (Logistique) |
|--------|------------------|----------------------|
| Sortie | Score brut | Probabilite [0, 1] |
| Objectif | Marge maximale | Maximum de vraisemblance |
| Gradient | Sparse (0 si bien classe) | Toujours non-nul |

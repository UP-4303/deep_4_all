# TP3 - Compte-rendu

Mazière Xavier
Luque Mattéo

## Hyperparamètres et architecture

Après multiples essais, nous sommes arrivés aux paramètres suivants :
```sh
uv run train_dungeon_logs.py --embed_dim 10 --hidden_dim 4 --num_layers 3 --dropout 0.3 --learning_rate 0.1 --mode lstm --bidirectional --epoch 50 --early_stopping --patience 5 --use_scheduler
```
```sh
uv run train_dungeon_logs.py --embed_dim 4 --hidden_dim 2 --num_layers 2 --dropout=0.17 --learning_rate=0.02 --mode lstm --bidirectional --epochs 50 --early_stopping --patience 5 --use_scheduler --optimizer adam
```

Avec ceci, nous avons obtenu une précision de 0.9727 sur la validation et 0.97 sur le test secret, pour un total de 7496 paramètres.

## Tentatives d'amélioration

Puisque le test secret semble contenir des logs globalement plus longs, une tentative de réduire le nombre de logs courts (catégorie `normal_short`) a été produite, mais cela a grandement réduit la qualité des résultats.

Pour ce même problème, une seconde approche a été d'ajouter des poids d'attention sur les logs appartenant à certaines catégories.

## Optimisation Automotisé

Après avoir fais un peu de recherche sur l'entrainement de réseau de neurone, nous avons découvert [Optuna](https://optuna.org/).
Il s'agit d'un outils permettant de rechercher de manière intéligente des configuration d'hyperparamètres. On définit un set de variable à explorer, un nombre entre deux bornes ou une valeur dans une liste par example, et on renvoie un score associé à ces paramètres.
Notre score est basé sur la validation accuracy du modèle entrainé avec plusieurs pénalités comme : avoir trop de paramètres, une trop grande différence entre la train accuracy et la validation accuracy, la variance de la validation accuracy.
Nous avons afiné la recherche en ignorant les modèles avec une validation accuracy trop basse afin de ne pas poluer la recherche.

## Résultat final

À la suite d'une longue recherche nous sommes arrivé à ces paramètres qui nous on donnée un modèle de 225 paramètres avec 90.47% de validation accuracy :
```sh
uv run train_dungeon_logs.py --embed_dim 1 --hidden_dim 1 --num_layers 1 --dropout 0.09091248360355031 --mode lstm --bidirectional --batch_size 32 --epoch 100 --learning_rate 0.002327067708383781 --weight_decay 0.0 --optimizer adam --early_stopping --patience 10 --use_scheduler
```

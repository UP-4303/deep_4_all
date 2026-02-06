# TP3 - Compte-rendu

Mazière Xavier
Luque Mattéo

## Hyperparamètres et architecture

Après multiples essais, nous sommes arrivés aux paramètres suivants :
`uv run train_dungeon_logs.py --embed_dim 10 --hidden_dim 4 --num_layers 3 --dropout 0.3 --learning_rate 0.1 --mode lstm --bidirectional --epoch 50 --early_stopping --patience 5 --use_scheduler`

Avec ceci, nous avons obtenu une précision de 0.9727 sur la validation et 0.97 sur le test secret, pour un total de 7496 paramètres.

## Tentatives d'amélioration

Puisque le test secret semble contenir des logs globalement plus longs, une tentative de réduire le nombre de logs courts (catégorie `normal_short`) a été produite, mais cela a grandement réduit la qualité des résultats.

Pour ce même problème, une seconde approche a été d'ajouter des poids d'attention sur les logs appartenant à certaines catégories.

# INFO905 - TP4 : Distillation de modèle

Mattéo LUQUE
Xavier MAZIERE

## Introduction

Dans ce TP, nous avons utilisé la méthode DASD (Distribution-Aligned Sequence Distillation) pour distiller un LLM de grande taille (GPT-OSS-120B) en un modèle plus petit (Qwen3-4B-Instruct-2507).

Cette méthode permet de transférer les connaissances d'un modèle enseignant vers un modèle étudiant en alignant les distributions de leurs sorties. Pour cela, les modèles sont interrogés sur deux ensembles de données.

Pour le premier ensemble de données, on interroge les modèles avec une faible température, et pour le second ensemble de données, on interroge les modèles avec une température plus élevée.

Les réponses sont ensuite filtrées pour ne conserver que les réponses où l'enseignant est suffisamment confiant et où l'étudiant est moins confiant que l'enseignant. Ces réponses sont ensuite utilisées pour entraîner l'étudiant.

Nous avons utilisé un ensemble de données de 1000 exemples de parties de poker Texas Hold'em, avec des transcriptions de parties et les actions correspondantes. Malgré des ajustements, le modèle enseignant ne semblait pas suffisamment confiant dans ses réponses à haute température, ce qui a limité la quantité de données utilisées pour l'entraînement.

## Méthodologie

## Résultats

## Discussion

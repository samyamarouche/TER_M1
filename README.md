# TER - Prédiction de Trajectoires GPS par LSTM

Université de Reims Champagne-Ardenne  
Master Informatique : IA 
Travail Encadré de Recherche (TER)  
Année universitaire 2024-2025

## Présentation

Ce travail de recherche a pour objectif d'explorer l'application de techniques d'intelligence artificielle à la prédiction de trajectoires de ballons météorologiques.

Plus précisément, il s'agira de :
- Étudier les approches classiques et actuelles de prédiction de trajectoire dans ce contexte
- Analyser comment des modèles d'intelligence artificielle, notamment de type séquentiel (RNN, LSTM), peuvent être adaptés à ce problème
- Mettre en œuvre un prototype de modèle d'intelligence artificielle simple pour prédire une trajectoire à partir de prévisions météorologiques.

L'objectif est d'explorer l'apprentissage profond pour la modélisation de séquences spatiales et temporelles, et d'évaluer la précision des prédictions sur des jeux de données réels.

## Structure du projet

```
TER/
├── Data/                  # Données brutes et prétraitées (NPY, CSV, etc.)
├── src/
│   ├── data/              # Scripts de chargement et préparation des données
│   └── model/             # Implémentation des modèles LSTM
├── main.py                # Script principal d'entraînement et d'évaluation
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier
```

## Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/samyamarouche/TER_M1.git
   cd TER
   ```

2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Préparer les données :**
   - Placer les fichiers de données GPS dans `Data/Simplified_Minute/` ou adapter le chemin dans `main.py`.

## Utilisation

Lancer l'entraînement et l'évaluation du modèle :

```bash
python main.py
```

Le script :
- Prépare les données (ou les recharge si elles existent déjà)
- Sépare les jeux d'entraînement et de test
- Entraîne un modèle LSTM avec apprentissage progressif
- Affiche les résultats et courbes de performance

## Détails techniques

- **Modèle :** LSTM simple avec couche dense de sortie (prédiction latitude/longitude)
- **Framework :** PyTorch
- **Fonctionnalités avancées :**
  - Apprentissage progressif (batch size, learning rate, dropout adaptatif)
  - Early stopping
  - Évaluation par MSE et distance Haversine

## Résultats attendus

- Prédiction précise de la prochaine position GPS à partir d'une fenêtre temporelle de positions précédentes.
- Visualisation des courbes de perte d'entraînement/validation.

## Auteurs

- Samy Amarouche
- Encadrant : Itheri Yahiaoui

## Licence

Projet académique - Usage pédagogique uniquement.


# Système de Trading Automatisé pour Bitget

Ce dépôt contient un système de trading automatisé complet pour les futures USDT sur Bitget. Le système utilise des indicateurs techniques avancés et des modèles de machine learning pour identifier les meilleures opportunités de trading et exécuter des ordres automatiquement.

## Fonctionnalités

- **Sélection automatique des paires** - Identifie dynamiquement les meilleures opportunités de trading
- **Stratégie basée sur des indicateurs techniques avancés** - Utilise MACD, RSI, Bollinger Bands, etc.
- **Intégration de machine learning** - Améliore les prédictions grâce à des modèles d'apprentissage automatique
- **Gestion du risque adaptative** - Ajuste les paramètres de trading en fonction des conditions du marché
- **Exécution directe via API** - Exécute les ordres directement sur Bitget
- **Interface Google Colab** - Interface utilisateur intuitive et accessible

## Structure du projet

```
bitget_trading_system/
├── code/
│   ├── pair_selection/       # Module de sélection des paires
│   ├── technical_indicators/ # Module d'indicateurs techniques
│   ├── ml_models/            # Module de modèles de machine learning
│   ├── risk_management/      # Module de gestion du risque
│   ├── api_integration/      # Module d'intégration API
│   ├── colab_interface/      # Interface Google Colab
│   └── optimized_parameters.py # Paramètres optimisés
├── docs/
│   ├── api_documentation.md  # Documentation de l'API Bitget
│   ├── strategy_requirements.md # Exigences de la stratégie
│   └── user_guide/           # Guide d'utilisation
│       ├── how_to_plug_and_play.md # Guide d'utilisation
│       └── technical_documentation.md # Documentation technique
├── tests/
│   ├── test_modules.py       # Tests unitaires
│   ├── test_integration.py   # Tests d'intégration
│   └── optimize_parameters.py # Script d'optimisation
└── setup.py                  # Script d'installation
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- Compte Bitget avec clés API

### Installation depuis GitHub

```bash
git clone https://github.com/username/bitget_trading_system.git
cd bitget_trading_system
pip install -e .
```

### Installation via Google Colab

Ouvrez le notebook `code/colab_interface/bitget_trading_system.ipynb` dans Google Colab et suivez les instructions.

## Guide de démarrage rapide

1. **Configurer les clés API Bitget**
   - Créez des clés API sur Bitget avec les permissions de trading
   - Configurez les clés dans le système

2. **Sélectionner les paires à trader**
   - Le système analysera toutes les paires disponibles
   - Il sélectionnera les meilleures opportunités

3. **Analyser les paires et générer des signaux**
   - Le système calculera les indicateurs techniques
   - Il générera des signaux de trading

4. **Calculer les paramètres de trading**
   - Le système déterminera la taille de position optimale
   - Il calculera les niveaux de stop loss et take profit

5. **Exécuter des ordres**
   - Le système exécutera des ordres sur Bitget
   - Il gérera les positions ouvertes

Pour des instructions détaillées, consultez le [Guide d'utilisation](docs/user_guide/how_to_plug_and_play.md).

## Documentation

- [Guide d'utilisation](docs/user_guide/how_to_plug_and_play.md) - Guide détaillé pour utiliser le système
- [Documentation technique](docs/user_guide/technical_documentation.md) - Documentation technique du système
- [Documentation de l'API Bitget](docs/api_documentation.md) - Documentation de l'API Bitget utilisée

## Tests

Pour exécuter les tests unitaires :

```bash
python -m unittest tests/test_modules.py
```

Pour exécuter les tests d'intégration :

```bash
python tests/test_integration.py
```

Pour optimiser les paramètres du système :

```bash
python tests/optimize_parameters.py
```

## Avertissement

Le trading de cryptomonnaies comporte des risques significatifs. Vous pouvez perdre tout ou partie de votre capital. Ce système ne garantit pas de profits et ne doit pas être utilisé avec des fonds que vous ne pouvez pas vous permettre de perdre.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contact

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub ou contacter l'équipe de support à support@example.com.

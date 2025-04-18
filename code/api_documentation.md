# Documentation de l'API Bitget pour le Trading de Futures USDT

## Introduction

Cette documentation résume les points clés de l'API Bitget pour le développement d'un système de trading automatisé pour les futures USDT sur Bitget.

## Types de Produits

Bitget propose trois types de produits pour le trading de futures :

| Type de Produit | Description |
| --- | --- |
| USDT-FUTURES | Futures USDT-M, réglés en USDT |
| USDC-FUTURES | Futures USDC-M, réglés en USDC |
| COIN-FUTURES | Futures Coin-M, réglés en cryptomonnaies |

Pour notre système, nous nous concentrerons sur **USDT-FUTURES**.

## Endpoints API Market Importants

### Get All Tickers
- **Endpoint**: GET /api/v2/mix/market/tickers
- **Description**: Récupère toutes les données de ticker pour un type de produit spécifié
- **Paramètres**:
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Essentiel pour la sélection automatique des paires à trader

### Get Candlestick Data
- **Endpoint**: GET /api/v2/mix/market/candles
- **Description**: Récupère les données de chandelier pour l'analyse technique
- **Règles de granularité**:
  - 1m, 3m, 5m: jusqu'à un mois
  - 15m: jusqu'à 52 jours
  - 30m: jusqu'à 62 jours
  - 1H: jusqu'à 83 jours
  - 2H: jusqu'à 120 jours
  - 4H: jusqu'à 240 jours
  - 6H: jusqu'à 360 jours
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Analyse technique et modèles de machine learning

### Get Open Interest
- **Endpoint**: GET /api/v2/mix/market/open-interest
- **Description**: Obtient le total des positions d'une paire de trading spécifique
- **Paramètres**:
  - symbol (String): Paire de trading
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Analyse du sentiment du marché et de la liquidité

### Get Current Funding Rate
- **Endpoint**: GET /api/v2/mix/market/current-funding-rate
- **Description**: Obtient le taux de financement actuel
- **Utilisation**: Optimisation des entrées/sorties de positions

## Sections à Explorer

Pour compléter notre système de trading automatisé, nous devons explorer les sections suivantes:

1. **Account** - Pour la gestion des comptes et des fonds
2. **Position** - Pour la gestion des positions ouvertes
3. **Trade** - Pour l'exécution des ordres
4. **Trigger Order** - Pour les ordres conditionnels
5. **Websocket** - Pour les mises à jour en temps réel

## Prochaines Étapes

1. Explorer les endpoints Account, Position et Trade
2. Documenter les paramètres d'authentification API
3. Comprendre les limites de taux et les restrictions
4. Étudier les exemples de code pour l'intégration API

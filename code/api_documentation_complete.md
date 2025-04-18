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

## Authentification API

Pour accéder à l'API Bitget, les en-têtes suivants sont nécessaires :

```
ACCESS-KEY: votre_api_key
ACCESS-SIGN: signature_générée
ACCESS-TIMESTAMP: timestamp_unix
ACCESS-PASSPHRASE: votre_passphrase
Content-Type: application/json
```

## Endpoints API Market Importants

### Get All Symbols
- **Endpoint**: GET /api/mix/v1/market/contracts
- **Description**: Récupère toutes les paires de trading disponibles
- **Paramètres**:
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Essentiel pour la sélection automatique des paires à trader

### Get All Tickers
- **Endpoint**: GET /api/mix/v1/market/tickers
- **Description**: Récupère toutes les données de ticker pour un type de produit spécifié
- **Paramètres**:
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Essentiel pour la sélection automatique des paires à trader

### Get Candlestick Data
- **Endpoint**: GET /api/mix/v1/market/candles
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
- **Endpoint**: GET /api/mix/v1/market/open-interest
- **Description**: Obtient le total des positions d'une paire de trading spécifique
- **Paramètres**:
  - symbol (String): Paire de trading
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 20 fois/1s (IP)
- **Utilisation**: Analyse du sentiment du marché et de la liquidité

### Get Current Funding Rate
- **Endpoint**: GET /api/mix/v1/market/current-funding-rate
- **Description**: Obtient le taux de financement actuel
- **Utilisation**: Optimisation des entrées/sorties de positions

## Endpoints API Account

### Get Single Account
- **Endpoint**: GET /api/mix/v1/account/account
- **Description**: Récupère les informations d'un compte spécifique
- **Paramètres**:
  - symbol (String): ID du symbole (doit être en majuscules)
  - marginCoin (String): Monnaie de marge
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Vérification du solde et des marges disponibles

### Get Account List
- **Endpoint**: GET /api/mix/v1/account/accounts
- **Description**: Récupère la liste de tous les comptes
- **Paramètres**:
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Vue d'ensemble de tous les comptes

### Change Leverage
- **Endpoint**: POST /api/mix/v1/account/set-leverage
- **Description**: Modifie l'effet de levier pour un symbole spécifique
- **Paramètres**:
  - symbol (String): ID du symbole
  - marginCoin (String): Monnaie de marge
  - leverage (String): Valeur de l'effet de levier
  - holdSide (String): Direction de la position (long/short)
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Ajustement du risque via l'effet de levier

### Change Margin Mode
- **Endpoint**: POST /api/mix/v1/account/set-margin-mode
- **Description**: Modifie le mode de marge (croisé ou isolé)
- **Paramètres**:
  - symbol (String): ID du symbole
  - marginCoin (String): Monnaie de marge
  - marginMode (String): Mode de marge (crossed/fixed)
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Gestion du risque via le mode de marge

## Endpoints API Trade

### Place Order
- **Endpoint**: POST /api/mix/v1/order/placeOrder
- **Description**: Place un ordre sur le marché
- **Paramètres**:
  - symbol (String): ID du symbole
  - marginCoin (String): Monnaie de marge
  - size (String): Taille de l'ordre
  - price (String): Prix de l'ordre (requis pour les ordres limites)
  - side (String): Direction (buy/sell)
  - orderType (String): Type d'ordre (limit/market)
  - timeInForceValue (String): Durée de validité (normal/postOnly/ioc/fok)
  - clientOid (String): ID client optionnel
  - reduceOnly (Boolean): Si l'ordre doit uniquement réduire la position
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Exécution des ordres d'achat et de vente

### Cancel Order
- **Endpoint**: POST /api/mix/v1/order/cancel-order
- **Description**: Annule un ordre existant
- **Paramètres**:
  - symbol (String): ID du symbole
  - orderId (String): ID de l'ordre à annuler
  - clientOid (String): ID client optionnel
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Annulation des ordres non exécutés

### Cancel All Orders
- **Endpoint**: POST /api/mix/v1/order/cancel-all-orders
- **Description**: Annule tous les ordres pour un symbole spécifique
- **Paramètres**:
  - symbol (String): ID du symbole
  - marginCoin (String): Monnaie de marge
  - productType (String): Type de produit (USDT-FUTURES)
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Annulation rapide de tous les ordres en cas de besoin

### Close All Position
- **Endpoint**: POST /api/mix/v1/position/close-all-position
- **Description**: Ferme toutes les positions ouvertes
- **Paramètres**:
  - productType (String): Type de produit (USDT-FUTURES)
  - marginCoin (String): Monnaie de marge
- **Limite de fréquence**: 10 fois/1s (uid)
- **Utilisation**: Fermeture d'urgence de toutes les positions

## Considérations Importantes

### Précision des Prix
- Les prix doivent respecter la précision définie pour chaque symbole
- Exemple: Pour BTCUSDT_UMCBL avec pricePlace=1 et priceEndStep=5, les prix valides sont 23455.0, 23455.5, 23446.0

### Limites d'Ordres
- Si la taille de l'ordre est supérieure à la taille maximale autorisée, une erreur sera renvoyée
- Erreur 40762: "The order size is greater than the max open size"

### Gestion des Positions
- Si la quantité de position est inférieure à la quantité minimale d'ordre, la position peut être fermée en entrant la quantité d'ordre dans la quantité restante de la position
- Si la quantité de position est supérieure à la quantité minimale d'ordre, la quantité minimale d'ordre doit être respectée lors de la fermeture de la position

## WebSocket API

Bitget propose également une API WebSocket pour les mises à jour en temps réel:

### Canaux Importants
- Ticker: Mises à jour des prix en temps réel
- Candle: Données de chandelier en temps réel
- Depth: Carnet d'ordres en temps réel
- Account: Mises à jour du compte
- Positions: Mises à jour des positions
- Orders: Mises à jour des ordres

### Format de Connexion
```json
{
  "op": "subscribe",
  "args": [
    {
      "instType": "MC",
      "channel": "ticker",
      "instId": "BTCUSDT"
    }
  ]
}
```

## Prochaines Étapes

1. Implémenter l'authentification API
2. Développer les fonctions de récupération des données de marché
3. Créer les fonctions d'exécution des ordres
4. Implémenter la gestion des positions et du risque
5. Intégrer les WebSockets pour les mises à jour en temps réel

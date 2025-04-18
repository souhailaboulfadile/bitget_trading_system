#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'optimisation des paramètres pour le système de trading automatisé Bitget.

Ce script utilise une recherche par grille pour trouver les meilleurs paramètres
pour les indicateurs techniques et les modèles de machine learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append('/home/ubuntu/bitget_trading_system')

# Import des modules à optimiser
from code.technical_indicators.indicators import IndicatorSet
from code.ml_models.models import ModelEnsemble
from code.api_integration.bitget_api import BitgetRESTClient


def fetch_historical_data(symbol, granularity, days_back):
    """
    Récupère les données historiques pour une paire spécifique.
    
    Args:
        symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
        granularity: Granularité des chandeliers (ex: 15m)
        days_back: Nombre de jours à récupérer
        
    Returns:
        DataFrame avec les données OHLCV
    """
    print(f"Récupération des données historiques pour {symbol}...")
    
    # Créer un client REST sans authentification pour les données publiques
    client = BitgetRESTClient()
    
    # Calculer le nombre de chandeliers à récupérer
    candles_per_day = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1
    }
    
    limit = min(1000, candles_per_day.get(granularity, 96) * days_back)
    
    # Récupérer les chandeliers
    candles = client.get_candles(symbol, granularity, limit)
    
    if not candles:
        print(f"Aucune donnée récupérée pour {symbol}")
        return pd.DataFrame()
    
    # Convertir en DataFrame
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convertir les types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    # Trier par timestamp
    df = df.sort_values("timestamp")
    
    print(f"Données récupérées: {len(df)} chandeliers du {df['timestamp'].min()} au {df['timestamp'].max()}")
    
    return df


def prepare_target_variable(df, forward_periods=3, threshold_pct=0.5):
    """
    Prépare la variable cible pour l'entraînement des modèles.
    
    Args:
        df: DataFrame avec les données OHLCV
        forward_periods: Nombre de périodes à regarder vers l'avant
        threshold_pct: Seuil de variation en pourcentage pour considérer un mouvement significatif
        
    Returns:
        DataFrame avec la variable cible ajoutée
    """
    print("Préparation de la variable cible...")
    
    # Calculer les rendements futurs
    df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
    
    # Créer la variable cible
    df['target'] = 0
    df.loc[df['future_return'] > threshold_pct/100, 'target'] = 1
    df.loc[df['future_return'] < -threshold_pct/100, 'target'] = -1
    
    # Supprimer les lignes avec des NaN
    df = df.dropna()
    
    # Afficher la distribution des classes
    class_counts = df['target'].value_counts()
    print(f"Distribution des classes: {class_counts.to_dict()}")
    
    return df


def optimize_indicator_parameters(df):
    """
    Optimise les paramètres des indicateurs techniques.
    
    Args:
        df: DataFrame avec les données OHLCV
        
    Returns:
        Dictionnaire des meilleurs paramètres
    """
    print("Optimisation des paramètres des indicateurs techniques...")
    
    # Paramètres à tester
    ema_periods = [10, 20, 30]
    sma_periods = [50, 100, 200]
    rsi_periods = [7, 14, 21]
    macd_params = [
        {"fast_period": 8, "slow_period": 17, "signal_period": 9},
        {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        {"fast_period": 16, "slow_period": 35, "signal_period": 9}
    ]
    bb_params = [
        {"period": 20, "std_dev": 2.0},
        {"period": 20, "std_dev": 2.5},
        {"period": 30, "std_dev": 2.0}
    ]
    
    # Préparer les résultats
    results = []
    
    # Tester toutes les combinaisons
    for ema in ema_periods:
        for sma in sma_periods:
            for rsi in rsi_periods:
                for macd in macd_params:
                    for bb in bb_params:
                        # Créer un IndicatorSet avec ces paramètres
                        indicator_set = IndicatorSet(
                            ema_period=ema,
                            sma_periods=[sma],
                            rsi_period=rsi,
                            macd_fast_period=macd["fast_period"],
                            macd_slow_period=macd["slow_period"],
                            macd_signal_period=macd["signal_period"],
                            bb_period=bb["period"],
                            bb_std_dev=bb["std_dev"]
                        )
                        
                        # Calculer les indicateurs
                        df_with_indicators = indicator_set.calculate_all(df)
                        
                        # Générer les signaux
                        signals = indicator_set.generate_signals(df_with_indicators)
                        combined_signal = indicator_set.generate_combined_signal(df_with_indicators)
                        
                        # Évaluer la performance
                        accuracy = accuracy_score(df_with_indicators['target'], combined_signal)
                        precision = precision_score(df_with_indicators['target'], combined_signal, average='macro', zero_division=0)
                        recall = recall_score(df_with_indicators['target'], combined_signal, average='macro', zero_division=0)
                        f1 = f1_score(df_with_indicators['target'], combined_signal, average='macro', zero_division=0)
                        
                        # Enregistrer les résultats
                        results.append({
                            "ema_period": ema,
                            "sma_period": sma,
                            "rsi_period": rsi,
                            "macd_fast_period": macd["fast_period"],
                            "macd_slow_period": macd["slow_period"],
                            "macd_signal_period": macd["signal_period"],
                            "bb_period": bb["period"],
                            "bb_std_dev": bb["std_dev"],
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        })
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    
    # Trier par f1_score
    results_df = results_df.sort_values("f1_score", ascending=False)
    
    # Afficher les meilleurs résultats
    print("\nMeilleurs paramètres pour les indicateurs techniques:")
    print(results_df.head(5))
    
    # Retourner les meilleurs paramètres
    best_params = results_df.iloc[0].to_dict()
    
    return best_params


def optimize_ml_models(df_with_indicators):
    """
    Optimise les paramètres des modèles de machine learning.
    
    Args:
        df_with_indicators: DataFrame avec les indicateurs techniques
        
    Returns:
        Dictionnaire des meilleurs paramètres
    """
    print("Optimisation des paramètres des modèles de machine learning...")
    
    # Créer un ModelEnsemble
    model_ensemble = ModelEnsemble()
    
    # Préparer les données
    X = df_with_indicators.drop(['timestamp', 'target', 'future_return'], axis=1, errors='ignore')
    y = df_with_indicators['target']
    
    # Définir les paramètres à tester pour le classificateur
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    # Définir la validation croisée par séries temporelles
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Créer le GridSearchCV
    grid_search = GridSearchCV(
        estimator=model_ensemble.classifier,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    # Exécuter la recherche par grille
    grid_search.fit(X, y)
    
    # Afficher les meilleurs paramètres
    print("\nMeilleurs paramètres pour le classificateur:")
    print(grid_search.best_params_)
    print(f"Meilleur score: {grid_search.best_score_:.4f}")
    
    # Retourner les meilleurs paramètres
    return grid_search.best_params_


def optimize_risk_parameters(df_with_indicators, combined_signal):
    """
    Optimise les paramètres de gestion du risque.
    
    Args:
        df_with_indicators: DataFrame avec les indicateurs techniques
        combined_signal: Série des signaux combinés
        
    Returns:
        Dictionnaire des meilleurs paramètres
    """
    print("Optimisation des paramètres de gestion du risque...")
    
    # Paramètres à tester
    risk_percentages = [0.005, 0.01, 0.02]
    stop_loss_atrs = [1.5, 2.0, 2.5]
    take_profit_ratios = [1.5, 2.0, 2.5]
    max_leverages = [3, 5, 10]
    
    # Préparer les résultats
    results = []
    
    # Simuler un compte de trading
    initial_balance = 10000.0
    
    # Tester toutes les combinaisons
    for risk_pct in risk_percentages:
        for sl_atr in stop_loss_atrs:
            for tp_ratio in take_profit_ratios:
                for max_lev in max_leverages:
                    # Simuler le trading
                    balance = initial_balance
                    positions = {}
                    
                    for i in range(1, len(df_with_indicators)):
                        date = df_with_indicators.index[i]
                        price = df_with_indicators['close'].iloc[i]
                        signal = combined_signal.iloc[i]
                        
                        # Calculer l'ATR
                        atr = df_with_indicators['high'].iloc[i-20:i].max() - df_with_indicators['low'].iloc[i-20:i].min()
                        atr = atr / 20
                        
                        # Gérer les positions existantes
                        for symbol, pos in list(positions.items()):
                            # Vérifier si le stop loss est atteint
                            if pos['type'] == 'long' and price <= pos['stop_loss']:
                                # Fermer la position avec perte
                                loss = (price / pos['entry_price'] - 1) * pos['size'] * pos['leverage']
                                balance += pos['size'] + loss
                                del positions[symbol]
                            elif pos['type'] == 'short' and price >= pos['stop_loss']:
                                # Fermer la position avec perte
                                loss = (1 - price / pos['entry_price']) * pos['size'] * pos['leverage']
                                balance += pos['size'] + loss
                                del positions[symbol]
                            # Vérifier si le take profit est atteint
                            elif pos['type'] == 'long' and price >= pos['take_profit']:
                                # Fermer la position avec profit
                                profit = (price / pos['entry_price'] - 1) * pos['size'] * pos['leverage']
                                balance += pos['size'] + profit
                                del positions[symbol]
                            elif pos['type'] == 'short' and price <= pos['take_profit']:
                                # Fermer la position avec profit
                                profit = (1 - price / pos['entry_price']) * pos['size'] * pos['leverage']
                                balance += pos['size'] + profit
                                del positions[symbol]
                            # Vérifier si le signal est inversé
                            elif (pos['type'] == 'long' and signal == -1) or (pos['type'] == 'short' and signal == 1):
                                # Fermer la position
                                if pos['type'] == 'long':
                                    pnl = (price / pos['entry_price'] - 1) * pos['size'] * pos['leverage']
                                else:
                                    pnl = (1 - price / pos['entry_price']) * pos['size'] * pos['leverage']
                                balance += pos['size'] + pnl
                                del positions[symbol]
                        
                        # Ouvrir de nouvelles positions
                        if signal != 0 and 'BTCUSDT' not in positions:
                            # Calculer la taille de la position
                            position_size = balance * risk_pct
                            
                            # Calculer le stop loss et le take profit
                            if signal == 1:  # Long
                                stop_loss = price - sl_atr * atr
                                take_profit = price + tp_ratio * sl_atr * atr
                                position_type = 'long'
                            else:  # Short
                                stop_loss = price + sl_atr * atr
                                take_profit = price - tp_ratio * sl_atr * atr
                                position_type = 'short'
                            
                            # Calculer le levier
                            leverage = min(max_lev, 1 / (price / stop_loss - 1) if signal == 1 else 1 / (1 - price / stop_loss))
                            
                            # Ouvrir la position
                            positions['BTCUSDT'] = {
                                'type': position_type,
                                'entry_price': price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'size': position_size,
                                'leverage': leverage
                            }
                            
                            # Déduire la taille de la position du solde
                            balance -= position_size
                    
                    # Fermer toutes les positions à la fin
                    final_balance = balance
                    for symbol, pos in positions.items():
                        price = df_with_indicators['close'].iloc[-1]
                        if pos['type'] == 'long':
                            pnl = (price / pos['entry_price'] - 1) * pos['size'] * pos['leverage']
                        else:
                            pnl = (1 - price / pos['entry_price']) * pos['size'] * pos['leverage']
                        final_balance += pos['size'] + pnl
                    
                    # Calculer les métriques
                    roi = (final_balance / initial_balance - 1) * 100
                    
                    # Enregistrer les résultats
                    results.append({
                        "risk_percentage": risk_pct,
                        "stop_loss_atr": sl_atr,
                        "take_profit_ratio": tp_ratio,
                        "max_leverage": max_lev,
                        "final_balance": final_balance,
                        "roi": roi
                    })
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    
    # Trier par ROI
    results_df = results_df.sort_values("roi", ascending=False)
    
    # Afficher les meilleurs résultats
    print("\nMeilleurs paramètres pour la gestion du risque:")
    print(results_df.head(5))
    
    # Retourner les meilleurs paramètres
    best_params = results_df.iloc[0].to_dict()
    
    return best_params


def main():
    """Fonction principale."""
    print("Démarrage de l'optimisation des paramètres...")
    
    # Récupérer les données historiques
    symbol = "BTCUSDT_UMCBL"
    granularity = "15m"
    days_back = 30
    
    df = fetch_historical_data(symbol, granularity, days_back)
    
    if df.empty:
        print("Impossible de continuer sans données.")
        return
    
    # Préparer la variable cible
    df = prepare_target_variable(df)
    
    # Optimiser les paramètres des indicateurs techniques
    indicator_params = optimize_indicator_parameters(df)
    
    # Créer un IndicatorSet avec les meilleurs paramètres
    indicator_set = IndicatorSet(
        ema_period=indicator_params["ema_period"],
        sma_periods=[indicator_params["sma_period"]],
        rsi_period=indicator_params["rsi_period"],
        macd_fast_period=indicator_params["macd_fast_period"],
        macd_slow_period=indicator_params["macd_slow_period"],
        macd_signal_period=indicator_params["macd_signal_period"],
        bb_period=indicator_params["bb_period"],
        bb_std_dev=indicator_params["bb_std_dev"]
    )
    
    # Calculer les indicateurs avec les meilleurs paramètres
    df_with_indicators = indicator_set.calculate_all(df)
    
    # Générer les signaux
    combined_signal = indicator_set.generate_combined_signal(df_with_indicators)
    
    # Optimiser les paramètres des modèles de machine learning
    ml_params = optimize_ml_models(df_with_indicators)
    
    # Optimiser les paramètres de gestion du risque
    risk_params = optimize_risk_parameters(df_with_indicators, combined_signal)
    
    # Afficher les meilleurs paramètres
    print("\nMeilleurs paramètres pour le système complet:")
    print("\nIndicateurs techniques:")
    for k, v in indicator_params.items():
        if k not in ["accuracy", "precision", "recall", "f1_score"]:
            print(f"  {k}: {v}")
    
    print("\nModèles de machine learning:")
    for k, v in ml_params.items():
        print(f"  {k}: {v}")
    
    print("\nGestion du risque:")
    for k, v in risk_params.items():
        if k not in ["final_balance", "roi"]:
            print(f"  {k}: {v}")
    
    print(f"\nPerformance attendue: ROI de {risk_params['roi']:.2f}% sur {days_back} jours")
    
    # Sauvegarder les meilleurs paramètres
    with open("/home/ubuntu/bitget_trading_system/code/optimized_parameters.py", "w") as f:
        f.write("#!/usr/bin/env python\n")
        f.write("# -*- coding: utf-8 -*-\n\n")
        f.write("\"\"\"Paramètres optimisés pour le système de trading automatisé Bitget.\"\"\"\n\n")
        
        f.write("# Paramètres des indicateurs techniques\n")
        f.write("INDICATOR_PARAMS = {\n")
        for k, v in indicator_params.items():
            if k not in ["accuracy", "precision", "recall", "f1_score"]:
                f.write(f"    \"{k}\": {v},\n")
        f.write("}\n\n")
        
        f.write("# Paramètres des modèles de machine learning\n")
        f.write("ML_PARAMS = {\n")
        for k, v in ml_params.items():
            f.write(f"    \"{k}\": {v},\n")
        f.write("}\n\n")
        
        f.write("# Paramètres de gestion du risque\n")
        f.write("RISK_PARAMS = {\n")
        for k, v in risk_params.items():
            if k not in ["final_balance", "roi"]:
                f.write(f"    \"{k}\": {v},\n")
        f.write("}\n")
    
    print("\nLes paramètres optimisés ont été sauvegardés dans /home/ubuntu/bitget_trading_system/code/optimized_parameters.py")


if __name__ == "__main__":
    main()

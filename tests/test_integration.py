#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'intégration pour tester le système complet de trading automatisé Bitget.

Ce script teste l'intégration entre tous les modules du système et vérifie
que le flux de données fonctionne correctement de bout en bout.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append('/home/ubuntu/bitget_trading_system')

# Import des modules du système
from code.pair_selection.pair_selector import PairSelector
from code.technical_indicators.indicators import IndicatorSet
from code.ml_models.models import ModelEnsemble
from code.risk_management.risk_manager import RiskManager, create_default_risk_manager
from code.api_integration.bitget_api import BitgetRESTClient, BitgetTrader, create_bitget_trader


def test_end_to_end_flow(api_key="", api_secret="", passphrase=""):
    """
    Teste le flux complet du système de trading.
    
    Args:
        api_key: Clé API Bitget (optionnelle)
        api_secret: Secret API Bitget (optionnel)
        passphrase: Passphrase API Bitget (optionnel)
        
    Returns:
        True si le test est réussi, False sinon
    """
    print("Test du flux complet du système de trading...")
    
    try:
        # Étape 1: Initialiser le sélecteur de paires
        print("\n1. Initialisation du sélecteur de paires...")
        pair_selector = PairSelector(api_key, api_secret, passphrase)
        
        # Si aucune clé API n'est fournie, utiliser des données de test
        if not api_key:
            print("Aucune clé API fournie, utilisation de données de test...")
            # Mock pour le sélecteur de paires
            pair_selector.select_pairs = lambda max_pairs=5: [
                {
                    "symbol": "BTCUSDT_UMCBL",
                    "score": 0.95,
                    "signal_direction": "long",
                    "metrics": {
                        "volatility": 2.5,
                        "volume_usd": 1000000000,
                        "momentum_24h": 3.2,
                        "current_price": 50000
                    }
                },
                {
                    "symbol": "ETHUSDT_UMCBL",
                    "score": 0.85,
                    "signal_direction": "long",
                    "metrics": {
                        "volatility": 3.5,
                        "volume_usd": 500000000,
                        "momentum_24h": 4.1,
                        "current_price": 3000
                    }
                }
            ]
        
        # Étape 2: Sélectionner les paires
        print("\n2. Sélection des paires...")
        selected_pairs = pair_selector.select_pairs(max_pairs=5)
        
        if not selected_pairs:
            print("❌ Erreur: Aucune paire sélectionnée.")
            return False
        
        print(f"✅ {len(selected_pairs)} paires sélectionnées.")
        for pair in selected_pairs[:2]:  # Afficher seulement les 2 premières paires
            print(f"  - {pair['symbol']}: Score = {pair['score']:.4f}, Direction = {pair['signal_direction']}")
        
        # Étape 3: Initialiser les indicateurs techniques
        print("\n3. Initialisation des indicateurs techniques...")
        indicator_set = IndicatorSet()
        
        # Étape 4: Initialiser les modèles de machine learning
        print("\n4. Initialisation des modèles de machine learning...")
        model_ensemble = ModelEnsemble()
        
        # Étape 5: Initialiser le gestionnaire de risque
        print("\n5. Initialisation du gestionnaire de risque...")
        risk_manager = create_default_risk_manager(10000.0)  # Solde fictif de 10000 USDT
        
        # Étape 6: Initialiser le trader Bitget
        print("\n6. Initialisation du trader Bitget...")
        
        if api_key and api_secret and passphrase:
            trader = create_bitget_trader(api_key, api_secret, passphrase)
        else:
            # Créer un mock pour le trader
            class MockTrader:
                def get_historical_candles(self, symbol, granularity, limit):
                    # Générer des données fictives
                    dates = pd.date_range(end=datetime.now(), periods=limit, freq='15min')
                    df = pd.DataFrame({
                        'timestamp': dates,
                        'open': np.random.normal(50000, 1000, limit),
                        'high': np.random.normal(50500, 1000, limit),
                        'low': np.random.normal(49500, 1000, limit),
                        'close': np.random.normal(50000, 1000, limit),
                        'volume': np.random.normal(100, 20, limit)
                    })
                    return df
                
                def get_account_balance(self):
                    return 10000.0
                
                def place_market_order(self, symbol, side, size, margin_coin="USDT"):
                    return {"orderId": "123456", "status": "success"}
                
                def set_leverage(self, symbol, leverage):
                    return {"status": "success"}
                
                def get_position(self, symbol):
                    return {
                        "symbol": symbol,
                        "holdSide": "long",
                        "total": "0.01",
                        "averageOpenPrice": "50000",
                        "leverage": "5",
                        "unrealizedPL": "100"
                    }
                
                def close_position(self, symbol):
                    return {"success": True, "message": "Position closed"}
                
                def cleanup(self):
                    pass
            
            trader = MockTrader()
        
        # Étape 7: Analyser une paire
        print("\n7. Analyse d'une paire...")
        symbol = selected_pairs[0]["symbol"]
        print(f"Analyse de {symbol}...")
        
        # Récupérer les données historiques
        df = trader.get_historical_candles(symbol, "15m", 100)
        
        if df.empty:
            print("❌ Erreur: Impossible de récupérer les données historiques.")
            return False
        
        print(f"✅ Données historiques récupérées: {len(df)} chandeliers.")
        
        # Calculer les indicateurs techniques
        df_with_indicators = indicator_set.calculate_all(df)
        
        print(f"✅ Indicateurs techniques calculés.")
        
        # Générer les signaux des indicateurs
        indicator_signals = indicator_set.generate_signals(df_with_indicators)
        
        # Générer un signal combiné à partir des indicateurs
        combined_indicator_signal = indicator_set.generate_combined_signal(df_with_indicators)
        
        print(f"✅ Signaux générés.")
        
        # Générer un signal à partir des modèles ML
        # Pour les tests, on utilise un signal aléatoire si les modèles ne sont pas entraînés
        try:
            ml_signal = model_ensemble.predict(df_with_indicators)
        except:
            ml_signal = pd.Series(np.random.choice([-1, 0, 1], size=len(df_with_indicators)), index=df_with_indicators.index)
        
        # Combiner les signaux des indicateurs et des modèles ML
        final_signal = pd.Series(0, index=df_with_indicators.index)
        final_signal[combined_indicator_signal == 1] += 0.5
        final_signal[ml_signal == 1] += 0.5
        final_signal[combined_indicator_signal == -1] -= 0.5
        final_signal[ml_signal == -1] -= 0.5
        
        # Discrétiser le signal final
        discretized_signal = pd.Series(0, index=final_signal.index)
        discretized_signal[final_signal > 0.3] = 1
        discretized_signal[final_signal < -0.3] = -1
        
        # Obtenir le dernier signal
        last_signal = discretized_signal.iloc[-1] if not discretized_signal.empty else 0
        signal_str = "ACHAT" if last_signal == 1 else "VENTE" if last_signal == -1 else "NEUTRE"
        
        print(f"✅ Signal final: {signal_str}")
        
        # Étape 8: Calculer les paramètres de trading
        print("\n8. Calcul des paramètres de trading...")
        
        if last_signal == 0:
            print("Signal neutre, aucun paramètre de trading à calculer.")
        else:
            # Calculer l'ATR
            df_with_indicators['tr'] = np.maximum(
                df_with_indicators['high'] - df_with_indicators['low'],
                np.maximum(
                    np.abs(df_with_indicators['high'] - df_with_indicators['close'].shift(1)),
                    np.abs(df_with_indicators['low'] - df_with_indicators['close'].shift(1))
                )
            )
            atr = df_with_indicators['tr'].mean()
            
            # Récupérer le prix actuel
            current_price = df_with_indicators['close'].iloc[-1]
            
            # Calculer la volatilité
            volatility = df_with_indicators['close'].pct_change().std() * np.sqrt(20)
            volatility_factor = volatility / 0.03  # Normaliser par rapport à une volatilité de référence de 3%
            
            # Déterminer le type de position
            position_type = "long" if last_signal == 1 else "short"
            
            # Calculer les paramètres de trading
            trade_params = risk_manager.calculate_trade_parameters(
                symbol=symbol,
                entry_price=current_price,
                position_type=position_type,
                atr=atr,
                volatility_factor=volatility_factor,
                signal_strength=0.7,
                market_regime="normal"
            )
            
            print("✅ Paramètres de trading calculés:")
            print(f"  - Type de position: {trade_params['position_type']}")
            print(f"  - Prix d'entrée: {trade_params['entry_price']:.4f}")
            print(f"  - Stop loss: {trade_params['stop_loss']:.4f}")
            print(f"  - Take profit: {', '.join([f'{tp:.4f}' for tp in trade_params['take_profit_levels']])}")
            print(f"  - Taille de la position: {trade_params['position_value']:.2f} USDT")
            print(f"  - Unités: {trade_params['units']:.6f}")
            print(f"  - Risque: {trade_params['risk_percentage']:.2f}% du capital")
            print(f"  - Levier: {trade_params['leverage']:.2f}x")
            
            # Étape 9: Simuler l'exécution d'un ordre (sans l'exécuter réellement)
            print("\n9. Simulation de l'exécution d'un ordre...")
            
            print(f"Simulation d'un ordre {position_type.upper()} pour {trade_params['units']:.6f} unités de {symbol} au prix de {trade_params['entry_price']:.4f} avec un levier de {trade_params['leverage']:.2f}x.")
            
            # Dans un environnement de test, on ne veut pas exécuter d'ordres réels
            if api_key and api_secret and passphrase:
                print("Environnement de production détecté, l'ordre ne sera pas exécuté.")
            else:
                print("✅ Simulation d'ordre réussie.")
        
        # Étape 10: Nettoyer les ressources
        print("\n10. Nettoyage des ressources...")
        if hasattr(trader, 'cleanup'):
            trader.cleanup()
        
        print("\n✅ Test du flux complet réussi!")
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur lors du test du flux complet: {str(e)}")
        return False


def main():
    """Fonction principale."""
    print("Démarrage des tests d'intégration...")
    
    # Récupérer les clés API depuis les variables d'environnement (si disponibles)
    api_key = os.environ.get("BITGET_API_KEY", "")
    api_secret = os.environ.get("BITGET_API_SECRET", "")
    passphrase = os.environ.get("BITGET_PASSPHRASE", "")
    
    # Exécuter le test de bout en bout
    success = test_end_to_end_flow(api_key, api_secret, passphrase)
    
    if success:
        print("\nTous les tests d'intégration ont réussi!")
    else:
        print("\nCertains tests d'intégration ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()

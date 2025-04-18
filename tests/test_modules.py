import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append('/home/ubuntu/bitget_trading_system')

# Import des modules à tester
from code.pair_selection.pair_selector import PairSelector, BitgetDataFetcher
from code.technical_indicators.indicators import IndicatorSet
from code.ml_models.models import ModelEnsemble
from code.risk_management.risk_manager import RiskManager
from code.api_integration.bitget_api import BitgetRESTClient, BitgetAuth


class TestPairSelector(unittest.TestCase):
    """Tests pour le module de sélection des paires."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer un mock pour BitgetDataFetcher
        self.mock_data_fetcher = MagicMock()
        
        # Configurer les données de test
        self.test_symbols = ["BTCUSDT_UMCBL", "ETHUSDT_UMCBL", "SOLUSDT_UMCBL"]
        self.test_tickers = [
            {"symbol": "BTCUSDT_UMCBL", "last": "50000", "high24h": "51000", "low24h": "49000", "volume24h": "1000", "priceChangePercent": "2.5"},
            {"symbol": "ETHUSDT_UMCBL", "last": "3000", "high24h": "3100", "low24h": "2900", "volume24h": "2000", "priceChangePercent": "3.5"},
            {"symbol": "SOLUSDT_UMCBL", "last": "100", "high24h": "110", "low24h": "90", "volume24h": "5000", "priceChangePercent": "5.0"}
        ]
        self.test_funding_rates = {
            "BTCUSDT_UMCBL": {"fundingRate": "0.0001"},
            "ETHUSDT_UMCBL": {"fundingRate": "0.0002"},
            "SOLUSDT_UMCBL": {"fundingRate": "0.0003"}
        }
        self.test_open_interests = {
            "BTCUSDT_UMCBL": {"amount": "10000000"},
            "ETHUSDT_UMCBL": {"amount": "5000000"},
            "SOLUSDT_UMCBL": {"amount": "1000000"}
        }
        
        # Configurer les retours du mock
        self.mock_data_fetcher.get_all_symbols.return_value = self.test_symbols
        self.mock_data_fetcher.get_tickers.return_value = self.test_tickers
        self.mock_data_fetcher.get_funding_rate.side_effect = lambda symbol: self.test_funding_rates.get(symbol, {"fundingRate": "0"})
        self.mock_data_fetcher.get_open_interest.side_effect = lambda symbol: self.test_open_interests.get(symbol, {"amount": "0"})
        
        # Créer l'instance de PairSelector avec le mock
        self.pair_selector = PairSelector(api_key="test", api_secret="test", passphrase="test")
        self.pair_selector.data_fetcher = self.mock_data_fetcher
    
    def test_calculate_metrics(self):
        """Teste le calcul des métriques pour une paire."""
        # Appeler la méthode à tester
        metrics = self.pair_selector._calculate_metrics(self.test_tickers[0])
        
        # Vérifier les résultats
        self.assertIn("volatility", metrics)
        self.assertIn("volume_usd", metrics)
        self.assertIn("momentum_24h", metrics)
        self.assertIn("current_price", metrics)
        
        # Vérifier les valeurs calculées
        self.assertAlmostEqual(metrics["current_price"], 50000.0)
        self.assertAlmostEqual(metrics["volatility"], 4.0)  # (51000 - 49000) / 50000 * 100
        self.assertAlmostEqual(metrics["momentum_24h"], 2.5)
    
    def test_select_pairs(self):
        """Teste la sélection des meilleures paires."""
        # Appeler la méthode à tester
        top_pairs = self.pair_selector.select_pairs(max_pairs=2)
        
        # Vérifier les résultats
        self.assertEqual(len(top_pairs), 2)
        self.assertEqual(top_pairs[0]["symbol"], "SOLUSDT_UMCBL")  # Devrait être le premier car meilleur momentum
        self.assertEqual(top_pairs[1]["symbol"], "ETHUSDT_UMCBL")  # Devrait être le second
        
        # Vérifier la structure des résultats
        for pair in top_pairs:
            self.assertIn("symbol", pair)
            self.assertIn("score", pair)
            self.assertIn("metrics", pair)
            self.assertIn("signal_direction", pair)


class TestIndicatorSet(unittest.TestCase):
    """Tests pour le module d'indicateurs techniques."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer des données de test
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)
        
        # Créer l'instance d'IndicatorSet
        self.indicator_set = IndicatorSet()
    
    def test_calculate_all(self):
        """Teste le calcul de tous les indicateurs."""
        # Appeler la méthode à tester
        df_with_indicators = self.indicator_set.calculate_all(self.test_data)
        
        # Vérifier que les indicateurs ont été ajoutés
        self.assertIn('ema_20', df_with_indicators.columns)
        self.assertIn('sma_50', df_with_indicators.columns)
        self.assertIn('sma_200', df_with_indicators.columns)
        self.assertIn('rsi', df_with_indicators.columns)
        self.assertIn('macd_line', df_with_indicators.columns)
        self.assertIn('macd_signal', df_with_indicators.columns)
        self.assertIn('macd_histogram', df_with_indicators.columns)
        self.assertIn('bb_upper', df_with_indicators.columns)
        self.assertIn('bb_middle', df_with_indicators.columns)
        self.assertIn('bb_lower', df_with_indicators.columns)
        
        # Vérifier que les calculs sont cohérents
        self.assertEqual(len(df_with_indicators), len(self.test_data))
        self.assertTrue(df_with_indicators['ema_20'].iloc[-1] > 0)
    
    def test_generate_signals(self):
        """Teste la génération de signaux à partir des indicateurs."""
        # Calculer les indicateurs
        df_with_indicators = self.indicator_set.calculate_all(self.test_data)
        
        # Appeler la méthode à tester
        signals = self.indicator_set.generate_signals(df_with_indicators)
        
        # Vérifier les résultats
        self.assertIn('ema_signal', signals)
        self.assertIn('macd_signal', signals)
        self.assertIn('rsi_signal', signals)
        self.assertIn('bb_signal', signals)
        
        # Vérifier que les signaux sont dans la plage attendue (-1, 0, 1)
        for signal_name in signals:
            unique_values = signals[signal_name].unique()
            for val in unique_values:
                self.assertIn(val, [-1, 0, 1])
    
    def test_generate_combined_signal(self):
        """Teste la génération d'un signal combiné."""
        # Calculer les indicateurs
        df_with_indicators = self.indicator_set.calculate_all(self.test_data)
        
        # Appeler la méthode à tester
        combined_signal = self.indicator_set.generate_combined_signal(df_with_indicators)
        
        # Vérifier les résultats
        self.assertEqual(len(combined_signal), len(self.test_data))
        
        # Vérifier que les signaux sont dans la plage attendue (-1, 0, 1)
        unique_values = combined_signal.unique()
        for val in unique_values:
            self.assertIn(val, [-1, 0, 1])


class TestModelEnsemble(unittest.TestCase):
    """Tests pour le module de modèles de machine learning."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer des données de test
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 100, 100),
            'ema_20': np.random.normal(100, 5, 100),
            'sma_50': np.random.normal(100, 5, 100),
            'rsi': np.random.normal(50, 10, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'macd_signal': np.random.normal(0, 1, 100),
            'macd_histogram': np.random.normal(0, 1, 100),
            'bb_upper': np.random.normal(110, 5, 100),
            'bb_middle': np.random.normal(100, 5, 100),
            'bb_lower': np.random.normal(90, 5, 100)
        }, index=dates)
        
        # Ajouter des labels pour l'entraînement
        self.test_data['target'] = np.random.choice([-1, 0, 1], size=100)
        
        # Créer l'instance de ModelEnsemble
        self.model_ensemble = ModelEnsemble()
    
    def test_train(self):
        """Teste l'entraînement des modèles."""
        # Appeler la méthode à tester
        self.model_ensemble.train(self.test_data, target_column='target')
        
        # Vérifier que les modèles ont été entraînés
        self.assertTrue(self.model_ensemble.is_trained)
        self.assertIsNotNone(self.model_ensemble.classifier)
        self.assertIsNotNone(self.model_ensemble.regressor)
    
    def test_predict(self):
        """Teste la prédiction avec les modèles."""
        # Entraîner les modèles
        self.model_ensemble.train(self.test_data, target_column='target')
        
        # Appeler la méthode à tester
        predictions = self.model_ensemble.predict(self.test_data)
        
        # Vérifier les résultats
        self.assertEqual(len(predictions), len(self.test_data))
        
        # Vérifier que les prédictions sont dans la plage attendue (-1, 0, 1)
        unique_values = predictions.unique()
        for val in unique_values:
            self.assertIn(val, [-1, 0, 1])


class TestRiskManager(unittest.TestCase):
    """Tests pour le module de gestion du risque."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer l'instance de RiskManager
        self.risk_manager = RiskManager(account_balance=10000.0)
    
    def test_calculate_trade_parameters(self):
        """Teste le calcul des paramètres de trading."""
        # Paramètres de test
        symbol = "BTCUSDT_UMCBL"
        entry_price = 50000.0
        position_type = "long"
        atr = 1000.0
        volatility_factor = 1.2
        signal_strength = 0.8
        market_regime = "normal"
        
        # Appeler la méthode à tester
        params = self.risk_manager.calculate_trade_parameters(
            symbol=symbol,
            entry_price=entry_price,
            position_type=position_type,
            atr=atr,
            volatility_factor=volatility_factor,
            signal_strength=signal_strength,
            market_regime=market_regime
        )
        
        # Vérifier les résultats
        self.assertEqual(params["symbol"], symbol)
        self.assertEqual(params["position_type"], position_type)
        self.assertEqual(params["entry_price"], entry_price)
        
        # Vérifier que les paramètres calculés sont cohérents
        self.assertGreater(params["position_value"], 0)
        self.assertGreater(params["units"], 0)
        self.assertGreater(params["leverage"], 1)
        self.assertLess(params["leverage"], 20)  # Levier raisonnable
        self.assertLess(params["stop_loss"], entry_price)  # Stop loss inférieur au prix d'entrée pour une position longue
        
        # Vérifier les niveaux de take profit
        for tp in params["take_profit_levels"]:
            self.assertGreater(tp, entry_price)  # Take profit supérieur au prix d'entrée pour une position longue
    
    def test_calculate_position_size(self):
        """Teste le calcul de la taille de position."""
        # Paramètres de test
        account_balance = 10000.0
        risk_percentage = 0.01  # 1%
        entry_price = 50000.0
        stop_loss = 49000.0
        
        # Appeler la méthode à tester
        position_value, units = self.risk_manager.position_sizing.calculate_position_size(
            account_balance=account_balance,
            risk_percentage=risk_percentage,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Vérifier les résultats
        expected_risk_amount = account_balance * risk_percentage
        expected_risk_per_unit = abs(entry_price - stop_loss)
        expected_units = expected_risk_amount / expected_risk_per_unit
        expected_position_value = expected_units * entry_price
        
        self.assertAlmostEqual(position_value, expected_position_value, delta=0.01)
        self.assertAlmostEqual(units, expected_units, delta=0.0001)


class TestBitgetAPI(unittest.TestCase):
    """Tests pour le module d'intégration API."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer des mocks pour les requêtes
        self.patcher = patch('code.api_integration.bitget_api.requests.Session')
        self.mock_session = self.patcher.start()
        
        # Configurer le mock pour simuler les réponses de l'API
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {"code": "00000", "data": [{"symbol": "BTCUSDT_UMCBL"}]}
        self.mock_response.raise_for_status.return_value = None
        self.mock_session.return_value.get.return_value = self.mock_response
        self.mock_session.return_value.post.return_value = self.mock_response
        
        # Créer l'instance de BitgetRESTClient
        self.client = BitgetRESTClient(api_key="test", api_secret="test", passphrase="test")
    
    def tearDown(self):
        """Nettoyage après les tests."""
        self.patcher.stop()
    
    def test_get_symbols(self):
        """Teste la récupération des symboles."""
        # Appeler la méthode à tester
        symbols = self.client.get_symbols()
        
        # Vérifier les résultats
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0]["symbol"], "BTCUSDT_UMCBL")
        
        # Vérifier que la requête a été effectuée correctement
        self.mock_session.return_value.get.assert_called_once()
    
    def test_place_order(self):
        """Teste le placement d'un ordre."""
        # Paramètres de test
        symbol = "BTCUSDT_UMCBL"
        margin_coin = "USDT"
        size = "0.001"
        side = "buy"
        order_type = "market"
        
        # Appeler la méthode à tester
        result = self.client.place_order(
            symbol=symbol,
            margin_coin=margin_coin,
            size=size,
            side=side,
            order_type=order_type
        )
        
        # Vérifier les résultats
        self.assertEqual(result, [{"symbol": "BTCUSDT_UMCBL"}])
        
        # Vérifier que la requête a été effectuée correctement
        self.mock_session.return_value.post.assert_called_once()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algorithme de sélection des paires pour le système de trading automatisé Bitget.

Ce module implémente un algorithme de classement multi-factoriel pour identifier
dynamiquement les meilleures opportunités de trading parmi toutes les paires
disponibles sur Bitget Futures USDT.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import requests
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pair_selector')

class BitgetDataFetcher:
    """
    Classe pour récupérer les données nécessaires à la sélection des paires depuis l'API Bitget.
    """
    
    BASE_URL = "https://api.bitget.com"
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = ""):
        """
        Initialise le fetcher de données Bitget.
        
        Args:
            api_key: Clé API Bitget (optionnelle pour les endpoints publics)
            api_secret: Secret API Bitget (optionnel pour les endpoints publics)
            passphrase: Passphrase API Bitget (optionnel pour les endpoints publics)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session = requests.Session()
    
    def _get_headers(self, timestamp: str = "", sign: str = "") -> Dict:
        """
        Génère les en-têtes pour les requêtes API.
        
        Args:
            timestamp: Timestamp pour l'authentification
            sign: Signature pour l'authentification
            
        Returns:
            Dict: En-têtes HTTP
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key and timestamp and sign:
            headers.update({
                "ACCESS-KEY": self.api_key,
                "ACCESS-SIGN": sign,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-PASSPHRASE": self.passphrase
            })
        
        return headers
    
    def get_all_symbols(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Récupère toutes les paires de trading disponibles.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            List[Dict]: Liste des paires disponibles avec leurs informations
        """
        endpoint = "/api/mix/v1/market/contracts"
        params = {"productType": product_type}
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if data["code"] == "00000":
                return data["data"]
            else:
                logger.error(f"API error: {data['msg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching symbols: {str(e)}")
            return []
    
    def get_all_tickers(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Récupère les données de ticker pour toutes les paires.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            List[Dict]: Liste des tickers avec leurs informations
        """
        endpoint = "/api/mix/v1/market/tickers"
        params = {"productType": product_type}
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if data["code"] == "00000":
                return data["data"]
            else:
                logger.error(f"API error: {data['msg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching tickers: {str(e)}")
            return []
    
    def get_candles(self, symbol: str, granularity: str = "15m", limit: int = 100) -> List[List]:
        """
        Récupère les données de chandeliers pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            granularity: Granularité des chandeliers (1m, 5m, 15m, etc.)
            limit: Nombre de chandeliers à récupérer
            
        Returns:
            List[List]: Liste des chandeliers [timestamp, open, high, low, close, volume]
        """
        endpoint = "/api/mix/v1/market/candles"
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if data["code"] == "00000":
                return data["data"]
            else:
                logger.error(f"API error: {data['msg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {str(e)}")
            return []
    
    def get_open_interest(self, symbol: str, product_type: str = "USDT-FUTURES") -> Dict:
        """
        Récupère l'intérêt ouvert pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Dict: Informations sur l'intérêt ouvert
        """
        endpoint = "/api/mix/v1/market/open-interest"
        params = {
            "symbol": symbol,
            "productType": product_type
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if data["code"] == "00000":
                return data["data"]
            else:
                logger.error(f"API error: {data['msg']}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {str(e)}")
            return {}
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Récupère le taux de financement actuel pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Dict: Informations sur le taux de financement
        """
        endpoint = "/api/mix/v1/market/current-funding-rate"
        params = {"symbol": symbol}
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if data["code"] == "00000":
                return data["data"]
            else:
                logger.error(f"API error: {data['msg']}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {str(e)}")
            return {}


class PairSelector:
    """
    Classe principale pour la sélection des paires à trader.
    
    Implémente un algorithme de classement multi-factoriel pour identifier
    dynamiquement les meilleures opportunités de trading.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = ""):
        """
        Initialise le sélecteur de paires.
        
        Args:
            api_key: Clé API Bitget
            api_secret: Secret API Bitget
            passphrase: Passphrase API Bitget
        """
        self.data_fetcher = BitgetDataFetcher(api_key, api_secret, passphrase)
        self.all_symbols = []
        self.all_tickers = []
        self.symbol_data = {}
        self.rankings = {}
        
        # Paramètres de sélection
        self.min_volume_usd = 5000000  # Volume minimum en USD sur 24h
        self.min_price_usd = 0.1  # Prix minimum en USD
        self.max_pairs = 10  # Nombre maximum de paires à sélectionner
        
        # Poids des facteurs pour le classement
        self.weights = {
            "volatility": 0.25,
            "volume": 0.20,
            "momentum": 0.25,
            "funding_rate": 0.15,
            "open_interest_change": 0.15
        }
    
    def fetch_market_data(self) -> None:
        """
        Récupère toutes les données de marché nécessaires pour la sélection des paires.
        """
        logger.info("Fetching market data...")
        
        # Récupérer toutes les paires disponibles
        self.all_symbols = self.data_fetcher.get_all_symbols()
        
        # Récupérer tous les tickers
        self.all_tickers = self.data_fetcher.get_all_tickers()
        
        # Créer un dictionnaire pour un accès facile aux données des tickers
        ticker_dict = {ticker["symbol"]: ticker for ticker in self.all_tickers}
        
        # Initialiser le dictionnaire des données des symboles
        self.symbol_data = {}
        
        # Pour chaque symbole, récupérer les données supplémentaires
        for symbol_info in self.all_symbols:
            symbol = symbol_info["symbol"]
            
            # Vérifier si le ticker existe pour ce symbole
            if symbol not in ticker_dict:
                continue
            
            # Récupérer les données de chandeliers
            candles_15m = self.data_fetcher.get_candles(symbol, "15m", 96)  # 24h de données
            candles_1h = self.data_fetcher.get_candles(symbol, "1h", 168)  # 7 jours de données
            
            if not candles_15m or not candles_1h:
                continue
            
            # Récupérer l'intérêt ouvert
            open_interest = self.data_fetcher.get_open_interest(symbol)
            
            # Récupérer le taux de financement
            funding_rate = self.data_fetcher.get_funding_rate(symbol)
            
            # Stocker toutes les données
            self.symbol_data[symbol] = {
                "info": symbol_info,
                "ticker": ticker_dict[symbol],
                "candles_15m": candles_15m,
                "candles_1h": candles_1h,
                "open_interest": open_interest,
                "funding_rate": funding_rate
            }
        
        logger.info(f"Fetched data for {len(self.symbol_data)} symbols")
    
    def calculate_metrics(self) -> None:
        """
        Calcule les métriques pour chaque paire à partir des données récupérées.
        """
        logger.info("Calculating metrics for each pair...")
        
        for symbol, data in self.symbol_data.items():
            try:
                # Convertir les chandeliers en DataFrame pour faciliter les calculs
                df_15m = pd.DataFrame(data["candles_15m"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df_15m = df_15m.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                
                df_1h = pd.DataFrame(data["candles_1h"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df_1h = df_1h.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                
                # Calculer la volatilité (ATR sur 14 périodes)
                df_15m["tr"] = np.maximum(
                    df_15m["high"] - df_15m["low"],
                    np.maximum(
                        np.abs(df_15m["high"] - df_15m["close"].shift(1)),
                        np.abs(df_15m["low"] - df_15m["close"].shift(1))
                    )
                )
                atr_15m = df_15m["tr"].rolling(14).mean().iloc[-1]
                
                # Normaliser l'ATR par le prix pour obtenir la volatilité en pourcentage
                current_price = float(data["ticker"]["last"])
                volatility = (atr_15m / current_price) * 100
                
                # Calculer le volume en USD sur 24h
                volume_usd = float(data["ticker"]["usdtVolume"])
                
                # Calculer le momentum (rendement sur 24h)
                momentum_24h = (float(data["ticker"]["last"]) / float(df_15m["close"].iloc[0]) - 1) * 100
                
                # Calculer le changement d'intérêt ouvert sur 24h (si disponible)
                open_interest_change = 0
                if data["open_interest"] and "amount" in data["open_interest"]:
                    # Nous n'avons pas l'historique de l'intérêt ouvert, donc nous utilisons une approximation
                    # basée sur le changement de prix et le volume
                    open_interest_amount = float(data["open_interest"]["amount"])
                    open_interest_change = (momentum_24h * volume_usd) / (open_interest_amount * 100) if open_interest_amount > 0 else 0
                
                # Obtenir le taux de financement (si disponible)
                funding_rate_value = 0
                if data["funding_rate"] and "fundingRate" in data["funding_rate"]:
                    funding_rate_value = float(data["funding_rate"]["fundingRate"]) * 100  # Convertir en pourcentage
                
                # Stocker les métriques calculées
                self.symbol_data[symbol]["metrics"] = {
                    "volatility": volatility,
                    "volume_usd": volume_usd,
                    "momentum_24h": momentum_24h,
                    "open_interest_change": open_interest_change,
                    "funding_rate": funding_rate_value,
                    "current_price": current_price
                }
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
                # Supprimer ce symbole des données si une erreur se produit
                self.symbol_data.pop(symbol, None)
        
        logger.info(f"Calculated metrics for {len(self.symbol_data)} symbols")
    
    def rank_pairs(self) -> List[Dict]:
        """
        Classe les paires selon les métriques calculées et les poids définis.
        
        Returns:
            List[Dict]: Liste des paires classées avec leurs scores
        """
        logger.info("Ranking pairs...")
        
        # Filtrer les paires selon les critères minimaux
        filtered_symbols = {}
        for symbol, data in self.symbol_data.items():
            if "metrics" not in data:
                continue
                
            metrics = data["metrics"]
            
            # Appliquer les filtres
            if metrics["volume_usd"] < self.min_volume_usd:
                continue
                
            if metrics["current_price"] < self.min_price_usd:
                continue
            
            filtered_symbols[symbol] = data
        
        logger.info(f"Filtered down to {len(filtered_symbols)} symbols")
        
        # Extraire les métriques pour la normalisation
        volatilities = [data["metrics"]["volatility"] for data in filtered_symbols.values()]
        volumes = [data["metrics"]["volume_usd"] for data in filtered_symbols.values()]
        momentums = [abs(data["metrics"]["momentum_24h"]) for data in filtered_symbols.values()]
        oi_changes = [abs(data["metrics"]["open_interest_change"]) for data in filtered_symbols.values()]
        funding_rates = [abs(data["metrics"]["funding_rate"]) for data in filtered_symbols.values()]
        
        # Normaliser les métriques (min-max scaling)
        def normalize(values):
            min_val = min(values) if values else 0
            max_val = max(values) if values else 1
            range_val = max_val - min_val
            if range_val == 0:
                return [0.5 for _ in values]
            return [(v - min_val) / range_val for v in values]
        
        norm_volatilities = normalize(volatilities)
        norm_volumes = normalize(volumes)
        norm_momentums = normalize(momentums)
        norm_oi_changes = normalize(oi_changes)
        norm_funding_rates = normalize(funding_rates)
        
        # Calculer les scores pour chaque paire
        scores = []
        for i, (symbol, data) in enumerate(filtered_symbols.items()):
            metrics = data["metrics"]
            
            # Calculer le score pondéré
            score = (
                self.weights["volatility"] * norm_volatilities[i] +
                self.weights["volume"] * norm_volumes[i] +
                self.weights["momentum"] * norm_momentums[i] +
                self.weights["open_interest_change"] * norm_oi_changes[i] +
                self.weights["funding_rate"] * norm_funding_rates[i]
            )
            
            # Déterminer la direction du signal basée sur le momentum
            signal_direction = "long" if metrics["momentum_24h"] > 0 else "short"
            
            # Ajouter à la liste des scores
            scores.append({
                "symbol": symbol,
                "score": score,
                "signal_direction": signal_direction,
                "metrics": metrics
            })
        
        # Trier les paires par score décroissant
        ranked_pairs = sorted(scores, key=lambda x: x["score"], reverse=True)
        
        # Limiter au nombre maximum de paires
        top_pairs = ranked_pairs[:self.max_pairs]
        
        logger.info(f"Selected top {len(top_pairs)} pairs")
        return top_pairs
    
    def select_pairs(self) -> List[Dict]:
        """
        Exécute le processus complet de sélection des paires.
        
        Returns:
            List[Dict]: Liste des meilleures paires à trader avec leurs informations
        """
        try:
            # Récupérer les données de marché
            self.fetch_market_data()
            
            # Calculer les métriques
            self.calculate_metrics()
            
            # Classer les paires
            ranked_pairs = self.rank_pairs()
            
            return ranked_pairs
            
        except Exception as e:
            logger.error(f"Error in pair selection process: {str(e)}")
            return []


def main():
    """Fonction principale pour tester le sélecteur de paires."""
    # Créer une instance du sélecteur de paires
    selector = PairSelector()
    
    # Sélectionner les meilleures paires
    top_pairs = selector.select_pairs()
    
    # Afficher les résultats
    print("\nTop pairs to trade:")
    print("-" * 80)
    print(f"{'Symbol':<15} {'Score':<10} {'Direction':<10} {'Price':<10} {'Vol 24h ($M)':<15} {'Volatility':<10} {'Momentum':<10}")
    print("-" * 80)
    
    for pair in top_pairs:
        metrics = pair["metrics"]
        print(f"{pair['symbol']:<15} {pair['score']:.4f}    {pair['signal_direction']:<10} "
              f"{metrics['current_price']:<10.4f} {metrics['volume_usd']/1000000:<15.2f} "
              f"{metrics['volatility']:<10.2f} {metrics['momentum_24h']:<10.2f}")


if __name__ == "__main__":
    main()

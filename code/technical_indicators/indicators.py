#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'indicateurs techniques avancés pour le système de trading automatisé Bitget.

Ce module implémente une variété d'indicateurs techniques utilisés pour l'analyse
des marchés financiers et la génération de signaux de trading.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum


class IndicatorCategory(Enum):
    """Catégories d'indicateurs techniques."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"


class TechnicalIndicator:
    """
    Classe de base pour tous les indicateurs techniques.
    """
    
    def __init__(self, name: str, category: IndicatorCategory):
        """
        Initialise un indicateur technique.
        
        Args:
            name: Nom de l'indicateur
            category: Catégorie de l'indicateur
        """
        self.name = name
        self.category = category
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'indicateur technique sur un DataFrame.
        
        Args:
            df: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame avec les colonnes de l'indicateur ajoutées
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur l'indicateur.
        
        Args:
            df: DataFrame contenant les données OHLCV et l'indicateur
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")


class MovingAverage(TechnicalIndicator):
    """
    Implémentation des moyennes mobiles (SMA, EMA, WMA, HMA).
    """
    
    def __init__(self, period: int = 14, ma_type: str = "sma"):
        """
        Initialise l'indicateur de moyenne mobile.
        
        Args:
            period: Période de la moyenne mobile
            ma_type: Type de moyenne mobile ('sma', 'ema', 'wma', 'hma')
        """
        super().__init__(f"{ma_type.upper()}({period})", IndicatorCategory.TREND)
        self.period = period
        self.ma_type = ma_type.lower()
        self.column_name = f"{self.ma_type}_{period}"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la moyenne mobile sur un DataFrame.
        
        Args:
            df: DataFrame contenant au moins une colonne 'close'
            
        Returns:
            DataFrame avec la colonne de moyenne mobile ajoutée
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        result_df = df.copy()
        
        if self.ma_type == "sma":
            result_df[self.column_name] = df['close'].rolling(window=self.period).mean()
        
        elif self.ma_type == "ema":
            result_df[self.column_name] = df['close'].ewm(span=self.period, adjust=False).mean()
        
        elif self.ma_type == "wma":
            weights = np.arange(1, self.period + 1)
            result_df[self.column_name] = df['close'].rolling(window=self.period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        
        elif self.ma_type == "hma":
            # Hull Moving Average
            half_period = int(self.period / 2)
            sqrt_period = int(np.sqrt(self.period))
            
            wma_half = df['close'].rolling(window=half_period).apply(
                lambda x: np.sum(np.arange(1, half_period + 1) * x) / np.arange(1, half_period + 1).sum(), raw=True
            )
            
            wma_full = df['close'].rolling(window=self.period).apply(
                lambda x: np.sum(np.arange(1, self.period + 1) * x) / np.arange(1, self.period + 1).sum(), raw=True
            )
            
            raw_hma = 2 * wma_half - wma_full
            
            result_df[self.column_name] = raw_hma.rolling(window=sqrt_period).apply(
                lambda x: np.sum(np.arange(1, sqrt_period + 1) * x) / np.arange(1, sqrt_period + 1).sum(), raw=True
            )
        
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur la moyenne mobile.
        
        Args:
            df: DataFrame contenant les données OHLCV et la moyenne mobile
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if self.column_name not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: prix au-dessus de la moyenne mobile
        signals[df['close'] > df[self.column_name]] = 1
        
        # Signal de vente: prix en-dessous de la moyenne mobile
        signals[df['close'] < df[self.column_name]] = -1
        
        return signals


class MACD(TechnicalIndicator):
    """
    Implémentation du MACD (Moving Average Convergence Divergence).
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialise l'indicateur MACD.
        
        Args:
            fast_period: Période de l'EMA rapide
            slow_period: Période de l'EMA lente
            signal_period: Période de l'EMA du signal
        """
        super().__init__(f"MACD({fast_period},{slow_period},{signal_period})", IndicatorCategory.TREND)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le MACD sur un DataFrame.
        
        Args:
            df: DataFrame contenant au moins une colonne 'close'
            
        Returns:
            DataFrame avec les colonnes MACD ajoutées
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        result_df = df.copy()
        
        # Calculer les EMAs
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculer la ligne MACD
        result_df['macd_line'] = fast_ema - slow_ema
        
        # Calculer la ligne de signal
        result_df['macd_signal'] = result_df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculer l'histogramme
        result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur le MACD.
        
        Args:
            df: DataFrame contenant les données OHLCV et le MACD
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: croisement à la hausse (MACD au-dessus de la ligne de signal)
        signals[(df['macd_line'] > df['macd_signal']) & 
                (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))] = 1
        
        # Signal de vente: croisement à la baisse (MACD en-dessous de la ligne de signal)
        signals[(df['macd_line'] < df['macd_signal']) & 
                (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))] = -1
        
        return signals


class RSI(TechnicalIndicator):
    """
    Implémentation du RSI (Relative Strength Index).
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialise l'indicateur RSI.
        
        Args:
            period: Période du RSI
            overbought: Niveau de surachat
            oversold: Niveau de survente
        """
        super().__init__(f"RSI({period})", IndicatorCategory.MOMENTUM)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le RSI sur un DataFrame.
        
        Args:
            df: DataFrame contenant au moins une colonne 'close'
            
        Returns:
            DataFrame avec la colonne RSI ajoutée
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        result_df = df.copy()
        
        # Calculer les variations de prix
        delta = result_df['close'].diff()
        
        # Séparer les variations positives et négatives
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculer la moyenne des gains et des pertes
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculer le RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculer le RSI
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur le RSI.
        
        Args:
            df: DataFrame contenant les données OHLCV et le RSI
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if 'rsi' not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: RSI sort de la zone de survente
        signals[(df['rsi'] > self.oversold) & (df['rsi'].shift(1) <= self.oversold)] = 1
        
        # Signal de vente: RSI entre dans la zone de surachat
        signals[(df['rsi'] > self.overbought) & (df['rsi'].shift(1) <= self.overbought)] = -1
        
        return signals


class BollingerBands(TechnicalIndicator):
    """
    Implémentation des bandes de Bollinger.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialise l'indicateur des bandes de Bollinger.
        
        Args:
            period: Période de la moyenne mobile
            std_dev: Nombre d'écarts-types pour les bandes
        """
        super().__init__(f"BB({period},{std_dev})", IndicatorCategory.VOLATILITY)
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les bandes de Bollinger sur un DataFrame.
        
        Args:
            df: DataFrame contenant au moins une colonne 'close'
            
        Returns:
            DataFrame avec les colonnes des bandes de Bollinger ajoutées
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        result_df = df.copy()
        
        # Calculer la moyenne mobile
        result_df['bb_middle'] = result_df['close'].rolling(window=self.period).mean()
        
        # Calculer l'écart-type
        rolling_std = result_df['close'].rolling(window=self.period).std()
        
        # Calculer les bandes supérieure et inférieure
        result_df['bb_upper'] = result_df['bb_middle'] + (rolling_std * self.std_dev)
        result_df['bb_lower'] = result_df['bb_middle'] - (rolling_std * self.std_dev)
        
        # Calculer la largeur des bandes (indicateur de volatilité)
        result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        
        # Calculer le %B (position relative du prix dans les bandes)
        result_df['bb_pct_b'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur les bandes de Bollinger.
        
        Args:
            df: DataFrame contenant les données OHLCV et les bandes de Bollinger
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: prix touche la bande inférieure
        signals[df['close'] <= df['bb_lower']] = 1
        
        # Signal de vente: prix touche la bande supérieure
        signals[df['close'] >= df['bb_upper']] = -1
        
        return signals


class ADX(TechnicalIndicator):
    """
    Implémentation de l'ADX (Average Directional Index).
    """
    
    def __init__(self, period: int = 14, threshold: int = 25):
        """
        Initialise l'indicateur ADX.
        
        Args:
            period: Période de l'ADX
            threshold: Seuil pour considérer une tendance comme forte
        """
        super().__init__(f"ADX({period})", IndicatorCategory.TREND)
        self.period = period
        self.threshold = threshold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'ADX sur un DataFrame.
        
        Args:
            df: DataFrame contenant les colonnes 'high', 'low', 'close'
            
        Returns:
            DataFrame avec les colonnes ADX ajoutées
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain a '{col}' column")
        
        result_df = df.copy()
        
        # Calculer les True Ranges
        result_df['tr'] = np.maximum(
            result_df['high'] - result_df['low'],
            np.maximum(
                np.abs(result_df['high'] - result_df['close'].shift(1)),
                np.abs(result_df['low'] - result_df['close'].shift(1))
            )
        )
        
        # Calculer les mouvements directionnels
        result_df['plus_dm'] = np.where(
            (result_df['high'] - result_df['high'].shift(1)) > (result_df['low'].shift(1) - result_df['low']),
            np.maximum(result_df['high'] - result_df['high'].shift(1), 0),
            0
        )
        
        result_df['minus_dm'] = np.where(
            (result_df['low'].shift(1) - result_df['low']) > (result_df['high'] - result_df['high'].shift(1)),
            np.maximum(result_df['low'].shift(1) - result_df['low'], 0),
            0
        )
        
        # Calculer les moyennes mobiles exponentielles
        result_df['tr_ema'] = result_df['tr'].ewm(span=self.period, adjust=False).mean()
        result_df['plus_di'] = 100 * (result_df['plus_dm'].ewm(span=self.period, adjust=False).mean() / result_df['tr_ema'])
        result_df['minus_di'] = 100 * (result_df['minus_dm'].ewm(span=self.period, adjust=False).mean() / result_df['tr_ema'])
        
        # Calculer le DX et l'ADX
        result_df['dx'] = 100 * np.abs(result_df['plus_di'] - result_df['minus_di']) / (result_df['plus_di'] + result_df['minus_di'])
        result_df['adx'] = result_df['dx'].ewm(span=self.period, adjust=False).mean()
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur l'ADX.
        
        Args:
            df: DataFrame contenant les données OHLCV et l'ADX
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if 'adx' not in df.columns or 'plus_di' not in df.columns or 'minus_di' not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: ADX > threshold et +DI > -DI
        signals[(df['adx'] > self.threshold) & (df['plus_di'] > df['minus_di'])] = 1
        
        # Signal de vente: ADX > threshold et -DI > +DI
        signals[(df['adx'] > self.threshold) & (df['minus_di'] > df['plus_di'])] = -1
        
        return signals


class Stochastic(TechnicalIndicator):
    """
    Implémentation de l'oscillateur stochastique.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3, 
                 overbought: float = 80, oversold: float = 20):
        """
        Initialise l'oscillateur stochastique.
        
        Args:
            k_period: Période du %K
            d_period: Période du %D
            slowing: Période de ralentissement
            overbought: Niveau de surachat
            oversold: Niveau de survente
        """
        super().__init__(f"Stoch({k_period},{d_period},{slowing})", IndicatorCategory.MOMENTUM)
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'oscillateur stochastique sur un DataFrame.
        
        Args:
            df: DataFrame contenant les colonnes 'high', 'low', 'close'
            
        Returns:
            DataFrame avec les colonnes stochastiques ajoutées
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain a '{col}' column")
        
        result_df = df.copy()
        
        # Calculer le %K
        lowest_low = result_df['low'].rolling(window=self.k_period).min()
        highest_high = result_df['high'].rolling(window=self.k_period).max()
        
        result_df['stoch_k_raw'] = 100 * ((result_df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Appliquer le ralentissement
        result_df['stoch_k'] = result_df['stoch_k_raw'].rolling(window=self.slowing).mean()
        
        # Calculer le %D
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=self.d_period).mean()
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère un signal de trading basé sur l'oscillateur stochastique.
        
        Args:
            df: DataFrame contenant les données OHLCV et l'oscillateur stochastique
            
        Returns:
            Series avec les signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
            df = self.calculate(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Signal d'achat: croisement à la hausse dans la zone de survente
        signals[(df['stoch_k'] > df['stoch_d']) & 
                (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & 
                (df['stoch_k'] < self.oversold)] = 1
        
        # Signal de vente: croisement à la baisse dans la zone de surachat
        signals[(df['stoch_k'] < df['stoch_d']) & 
                (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & 
                (df['stoch_k'] > self.overbought)] = -1
        
        return signals


class ATR(TechnicalIndicator):
    """
    Implémentation de l'ATR (Average True Range).
    """
    
    def __init__(self, period: int = 14):
        """
        Initialise l'indicateur ATR.
        
        Args:
            period: Période de l'ATR
        """
        super().__init__(f"ATR({period})", IndicatorCategory.VOLATILITY)
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'ATR sur un DataFrame.
        
        Args:
            df: DataFrame contenant les colonnes 'high', 'low', 'close'
            
        Returns:
            DataFrame avec la colonne ATR ajoutée
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain a '{col}' column")
        
        result_df = df.copy()
        
        # Calculer les True Ranges
        result_df['tr'] = np.maximum(
            result_df['high'] - result_df['low'],
            np.maximum(
                np.abs(result_df['high'] - result_df['close'].shift(1)),
                np.abs(result_df['low'] - result_df['close'].shift(1))
            )
        )
        
        # Calculer l'ATR
        result_df['atr'] = result_df['tr'].rolling(window=self.period).mean()
        
        return result_df
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        L'ATR est un indicateur de volatilité et ne génère pas directement de signaux.
        Cette méthode retourne une série de zéros.
        
        Args:
            df: DataFrame contenant les données OHLCV et l'ATR
            
        Returns:
            Series avec les signaux (toujours 0)
        """
        if 'atr' not in df.columns:
            df = self.calculate(df)
        
        return pd.Series(0, index=df.index)


class IndicatorFactory:
    """
    Factory pour créer des instances d'indicateurs techniques.
    """
    
    @staticmethod
    def create_indicator(indicator_type: str, **kwargs) -> TechnicalIndicator:
        """
        Crée une instance d'indicateur technique.
        
        Args:
            indicator_type: Type d'indicateur ('ma', 'macd', 'rsi', 'bb', 'adx', 'stoch', 'atr')
            **kwargs: Paramètres spécifiques à l'indicateur
            
        Returns:
            Instance de TechnicalIndicator
        """
        indicator_type = indicator_type.lower()
        
        if indicator_type == 'ma':
            return MovingAverage(**kwargs)
        
        elif indicator_type == 'macd':
            return MACD(**kwargs)
        
        elif indicator_type == 'rsi':
            return RSI(**kwargs)
        
        elif indicator_type == 'bb':
            return BollingerBands(**kwargs)
        
        elif indicator_type == 'adx':
            return ADX(**kwargs)
        
        elif indicator_type == 'stoch':
            return Stochastic(**kwargs)
        
        elif indicator_type == 'atr':
            return ATR(**kwargs)
        
        else:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")


class IndicatorSet:
    """
    Ensemble d'indicateurs techniques pour l'analyse complète d'un actif.
    """
    
    def __init__(self):
        """
        Initialise un ensemble d'indicateurs techniques.
        """
        self.indicators = []
    
    def add_indicator(self, indicator: TechnicalIndicator) -> None:
        """
        Ajoute un indicateur à l'ensemble.
        
        Args:
            indicator: Instance de TechnicalIndicator
        """
        self.indicators.append(indicator)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les indicateurs sur un DataFrame.
        
        Args:
            df: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame avec toutes les colonnes d'indicateurs ajoutées
        """
        result_df = df.copy()
        
        for indicator in self.indicators:
            result_df = indicator.calculate(result_df)
        
        return result_df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Génère les signaux pour tous les indicateurs.
        
        Args:
            df: DataFrame contenant les données OHLCV et les indicateurs
            
        Returns:
            Dictionnaire de Series avec les signaux pour chaque indicateur
        """
        signals = {}
        
        for indicator in self.indicators:
            signals[indicator.name] = indicator.generate_signal(df)
        
        return signals
    
    def generate_combined_signal(self, df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Génère un signal combiné à partir de tous les indicateurs.
        
        Args:
            df: DataFrame contenant les données OHLCV et les indicateurs
            weights: Dictionnaire des poids pour chaque indicateur (optionnel)
            
        Returns:
            Series avec les signaux combinés
        """
        if not self.indicators:
            return pd.Series(0, index=df.index)
        
        # Générer les signaux individuels
        signals = self.generate_signals(df)
        
        # Si aucun poids n'est fourni, utiliser des poids égaux
        if weights is None:
            weights = {indicator.name: 1.0 / len(self.indicators) for indicator in self.indicators}
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculer le signal combiné pondéré
        combined_signal = pd.Series(0.0, index=df.index)
        
        for indicator in self.indicators:
            if indicator.name in normalized_weights:
                combined_signal += signals[indicator.name] * normalized_weights[indicator.name]
        
        # Discrétiser le signal combiné
        discretized_signal = pd.Series(0, index=df.index)
        discretized_signal[combined_signal > 0.2] = 1
        discretized_signal[combined_signal < -0.2] = -1
        
        return discretized_signal


def create_default_indicator_set() -> IndicatorSet:
    """
    Crée un ensemble d'indicateurs par défaut pour l'analyse technique.
    
    Returns:
        IndicatorSet: Ensemble d'indicateurs par défaut
    """
    indicator_set = IndicatorSet()
    
    # Ajouter des moyennes mobiles
    indicator_set.add_indicator(MovingAverage(period=20, ma_type="ema"))
    indicator_set.add_indicator(MovingAverage(period=50, ma_type="ema"))
    indicator_set.add_indicator(MovingAverage(period=200, ma_type="sma"))
    
    # Ajouter le MACD
    indicator_set.add_indicator(MACD(fast_period=12, slow_period=26, signal_period=9))
    
    # Ajouter le RSI
    indicator_set.add_indicator(RSI(period=14, overbought=70, oversold=30))
    
    # Ajouter les bandes de Bollinger
    indicator_set.add_indicator(BollingerBands(period=20, std_dev=2.0))
    
    # Ajouter l'ADX
    indicator_set.add_indicator(ADX(period=14, threshold=25))
    
    # Ajouter l'oscillateur stochastique
    indicator_set.add_indicator(Stochastic(k_period=14, d_period=3, slowing=3))
    
    # Ajouter l'ATR
    indicator_set.add_indicator(ATR(period=14))
    
    return indicator_set


if __name__ == "__main__":
    # Exemple d'utilisation
    import yfinance as yf
    
    # Télécharger des données historiques
    data = yf.download("BTC-USD", period="1y", interval="1d")
    
    # Créer un ensemble d'indicateurs
    indicator_set = create_default_indicator_set()
    
    # Calculer tous les indicateurs
    data_with_indicators = indicator_set.calculate_all(data)
    
    # Générer les signaux
    signals = indicator_set.generate_signals(data_with_indicators)
    
    # Générer un signal combiné
    combined_signal = indicator_set.generate_combined_signal(data_with_indicators)
    
    # Afficher les dernières lignes
    print(data_with_indicators.tail())
    print("\nCombined Signal:")
    print(combined_signal.tail())

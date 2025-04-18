"""
Module d'indicateurs techniques pour le système de trading automatisé Bitget.

Ce module contient les classes et fonctions nécessaires pour calculer
divers indicateurs techniques et générer des signaux de trading.
"""

from .indicators import (
    TechnicalIndicator, IndicatorCategory, MovingAverage, MACD, RSI,
    BollingerBands, ADX, Stochastic, ATR, IndicatorFactory,
    IndicatorSet, create_default_indicator_set
)

__all__ = [
    'TechnicalIndicator', 'IndicatorCategory', 'MovingAverage', 'MACD', 
    'RSI', 'BollingerBands', 'ADX', 'Stochastic', 'ATR', 
    'IndicatorFactory', 'IndicatorSet', 'create_default_indicator_set'
]

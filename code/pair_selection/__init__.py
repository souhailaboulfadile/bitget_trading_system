"""
Module de sélection des paires pour le système de trading automatisé Bitget.

Ce module contient les classes et fonctions nécessaires pour identifier
dynamiquement les meilleures opportunités de trading parmi toutes les paires
disponibles sur Bitget Futures USDT.
"""

from .pair_selector import BitgetDataFetcher, PairSelector

__all__ = ['BitgetDataFetcher', 'PairSelector']

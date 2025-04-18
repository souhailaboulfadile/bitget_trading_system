"""
Module d'intégration API pour le système de trading automatisé Bitget.

Ce module contient les classes et fonctions nécessaires pour interagir avec l'API
de Bitget, y compris l'authentification, la récupération des données de marché,
et l'exécution des ordres.
"""

from .bitget_api import (
    BitgetAPIError, BitgetAuth, BitgetRESTClient, BitgetWebSocketClient,
    BitgetTrader, create_bitget_trader
)

__all__ = [
    'BitgetAPIError', 'BitgetAuth', 'BitgetRESTClient', 'BitgetWebSocketClient',
    'BitgetTrader', 'create_bitget_trader'
]

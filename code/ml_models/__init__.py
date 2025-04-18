"""
Module de modèles de machine learning pour le système de trading automatisé Bitget.

Ce module contient les classes et fonctions nécessaires pour la prédiction
des mouvements de prix et la génération de signaux de trading.
"""

from .models import (
    ModelType, FeatureEngineering, BaseModel, DirectionClassifier,
    MarketRegimeClusterer, ModelEnsemble, create_default_model_ensemble
)

__all__ = [
    'ModelType', 'FeatureEngineering', 'BaseModel', 'DirectionClassifier',
    'MarketRegimeClusterer', 'ModelEnsemble', 'create_default_model_ensemble'
]

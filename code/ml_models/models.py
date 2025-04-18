#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de modèles de machine learning pour le système de trading automatisé Bitget.

Ce module implémente différents modèles de machine learning pour la prédiction
des mouvements de prix et la génération de signaux de trading.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import xgboost as xgb

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_models')


class ModelType(Enum):
    """Types de modèles de machine learning."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    REINFORCEMENT = "reinforcement"


class FeatureEngineering:
    """
    Classe pour la création et la transformation des caractéristiques (features)
    à partir des données de marché et des indicateurs techniques.
    """
    
    def __init__(self):
        """
        Initialise le processeur de caractéristiques.
        """
        self.scalers = {}
    
    def create_features(self, df: pd.DataFrame, include_indicators: bool = True) -> pd.DataFrame:
        """
        Crée des caractéristiques à partir des données brutes et des indicateurs techniques.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            DataFrame avec les caractéristiques créées
        """
        if df.empty:
            return pd.DataFrame()
        
        # Copier le DataFrame pour éviter de modifier l'original
        features_df = pd.DataFrame(index=df.index)
        
        # Caractéristiques basées sur les prix
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Rendements
            features_df['return_1d'] = df['close'].pct_change(1)
            features_df['return_2d'] = df['close'].pct_change(2)
            features_df['return_5d'] = df['close'].pct_change(5)
            features_df['return_10d'] = df['close'].pct_change(10)
            
            # Volatilité
            features_df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
            features_df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
            features_df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
            
            # Ratios de prix
            features_df['close_to_open'] = df['close'] / df['open']
            features_df['high_to_low'] = df['high'] / df['low']
            features_df['close_to_high'] = df['close'] / df['high']
            features_df['close_to_low'] = df['close'] / df['low']
            
            # Moyennes mobiles
            features_df['ma_5d'] = df['close'].rolling(5).mean() / df['close']
            features_df['ma_10d'] = df['close'].rolling(10).mean() / df['close']
            features_df['ma_20d'] = df['close'].rolling(20).mean() / df['close']
            features_df['ma_50d'] = df['close'].rolling(50).mean() / df['close']
            
            # Écarts-types
            features_df['std_5d'] = df['close'].rolling(5).std() / df['close']
            features_df['std_10d'] = df['close'].rolling(10).std() / df['close']
            features_df['std_20d'] = df['close'].rolling(20).std() / df['close']
        
        # Caractéristiques basées sur le volume
        if 'volume' in df.columns:
            features_df['volume_change_1d'] = df['volume'].pct_change(1)
            features_df['volume_change_5d'] = df['volume'].pct_change(5)
            features_df['volume_ma_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
            features_df['volume_ma_ratio_10d'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # Inclure les indicateurs techniques si demandé
        if include_indicators:
            # MACD
            if all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
                features_df['macd_line'] = df['macd_line']
                features_df['macd_signal'] = df['macd_signal']
                features_df['macd_histogram'] = df['macd_histogram']
                features_df['macd_cross'] = np.where(
                    df['macd_line'] > df['macd_signal'], 1, 
                    np.where(df['macd_line'] < df['macd_signal'], -1, 0)
                )
            
            # RSI
            if 'rsi' in df.columns:
                features_df['rsi'] = df['rsi']
                features_df['rsi_change'] = df['rsi'].diff(1)
                features_df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
                features_df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            
            # Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct_b']):
                features_df['bb_width'] = df['bb_width']
                features_df['bb_pct_b'] = df['bb_pct_b']
                features_df['bb_upper_touch'] = np.where(df['close'] >= df['bb_upper'], 1, 0)
                features_df['bb_lower_touch'] = np.where(df['close'] <= df['bb_lower'], 1, 0)
            
            # ADX
            if all(col in df.columns for col in ['adx', 'plus_di', 'minus_di']):
                features_df['adx'] = df['adx']
                features_df['plus_di'] = df['plus_di']
                features_df['minus_di'] = df['minus_di']
                features_df['adx_trend_strength'] = np.where(df['adx'] > 25, 1, 0)
                features_df['adx_trend_direction'] = np.where(
                    df['plus_di'] > df['minus_di'], 1, 
                    np.where(df['plus_di'] < df['minus_di'], -1, 0)
                )
            
            # Stochastic
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                features_df['stoch_k'] = df['stoch_k']
                features_df['stoch_d'] = df['stoch_d']
                features_df['stoch_cross'] = np.where(
                    df['stoch_k'] > df['stoch_d'], 1, 
                    np.where(df['stoch_k'] < df['stoch_d'], -1, 0)
                )
                features_df['stoch_overbought'] = np.where(df['stoch_k'] > 80, 1, 0)
                features_df['stoch_oversold'] = np.where(df['stoch_k'] < 20, 1, 0)
        
        # Supprimer les lignes avec des valeurs NaN
        features_df = features_df.dropna()
        
        return features_df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
        """
        Crée une variable cible pour la classification ou la régression.
        
        Args:
            df: DataFrame contenant au moins une colonne 'close'
            horizon: Horizon de prédiction en périodes
            threshold: Seuil pour la classification (en pourcentage)
            
        Returns:
            Series avec la variable cible
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        # Calculer le rendement futur
        future_return = df['close'].pct_change(horizon).shift(-horizon)
        
        # Pour la classification, convertir en classes (-1, 0, 1)
        if threshold > 0:
            target = pd.Series(0, index=df.index)
            target[future_return > threshold] = 1
            target[future_return < -threshold] = -1
        else:
            # Pour la régression, utiliser directement le rendement
            target = future_return
        
        return target
    
    def scale_features(self, features: pd.DataFrame, scaler_name: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """
        Normalise les caractéristiques.
        
        Args:
            features: DataFrame contenant les caractéristiques
            scaler_name: Nom du scaler ('standard' ou 'minmax')
            fit: Si True, ajuste le scaler aux données, sinon utilise un scaler existant
            
        Returns:
            DataFrame avec les caractéristiques normalisées
        """
        if features.empty:
            return pd.DataFrame()
        
        # Créer un nouveau scaler si nécessaire
        if fit or scaler_name not in self.scalers:
            if scaler_name == 'standard':
                self.scalers[scaler_name] = StandardScaler()
            elif scaler_name == 'minmax':
                self.scalers[scaler_name] = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaler: {scaler_name}")
        
        # Appliquer le scaler
        if fit:
            scaled_values = self.scalers[scaler_name].fit_transform(features)
        else:
            scaled_values = self.scalers[scaler_name].transform(features)
        
        # Créer un nouveau DataFrame avec les valeurs normalisées
        scaled_features = pd.DataFrame(
            scaled_values, 
            index=features.index, 
            columns=features.columns
        )
        
        return scaled_features
    
    def save_scalers(self, directory: str) -> None:
        """
        Sauvegarde les scalers dans un répertoire.
        
        Args:
            directory: Chemin du répertoire où sauvegarder les scalers
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(directory, f"{name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} scaler to {scaler_path}")
    
    def load_scalers(self, directory: str) -> None:
        """
        Charge les scalers depuis un répertoire.
        
        Args:
            directory: Chemin du répertoire contenant les scalers
        """
        for scaler_name in ['standard', 'minmax']:
            scaler_path = os.path.join(directory, f"{scaler_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers[scaler_name] = joblib.load(scaler_path)
                logger.info(f"Loaded {scaler_name} scaler from {scaler_path}")


class BaseModel:
    """
    Classe de base pour tous les modèles de machine learning.
    """
    
    def __init__(self, name: str, model_type: ModelType):
        """
        Initialise un modèle de base.
        
        Args:
            name: Nom du modèle
            model_type: Type du modèle
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.feature_engineering = FeatureEngineering()
        self.trained = False
    
    def train(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Entraîne le modèle sur les données.
        
        Args:
            df: DataFrame contenant les données
            **kwargs: Paramètres supplémentaires pour l'entraînement
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions avec le modèle.
        
        Args:
            df: DataFrame contenant les données
            
        Returns:
            Array des prédictions
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Évalue les performances du modèle.
        
        Args:
            df: DataFrame contenant les données
            **kwargs: Paramètres supplémentaires pour l'évaluation
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save(self, directory: str) -> None:
        """
        Sauvegarde le modèle dans un répertoire.
        
        Args:
            directory: Chemin du répertoire où sauvegarder le modèle
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained, nothing to save")
            return
        
        os.makedirs(directory, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = os.path.join(directory, f"{self.name}_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Sauvegarder les scalers
        scalers_dir = os.path.join(directory, f"{self.name}_scalers")
        self.feature_engineering.save_scalers(scalers_dir)
        
        logger.info(f"Saved model {self.name} to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Charge le modèle depuis un répertoire.
        
        Args:
            directory: Chemin du répertoire contenant le modèle
        """
        model_path = os.path.join(directory, f"{self.name}_model.joblib")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found")
            return
        
        # Charger le modèle
        self.model = joblib.load(model_path)
        
        # Charger les scalers
        scalers_dir = os.path.join(directory, f"{self.name}_scalers")
        if os.path.exists(scalers_dir):
            self.feature_engineering.load_scalers(scalers_dir)
        
        self.trained = True
        logger.info(f"Loaded model {self.name} from {directory}")


class DirectionClassifier(BaseModel):
    """
    Modèle de classification pour prédire la direction du prix.
    """
    
    def __init__(self, name: str = "direction_classifier", classifier_type: str = "random_forest"):
        """
        Initialise un classifieur de direction.
        
        Args:
            name: Nom du modèle
            classifier_type: Type de classifieur ('random_forest', 'gradient_boosting', 'logistic', 'svm', 'mlp', 'xgboost')
        """
        super().__init__(name, ModelType.CLASSIFICATION)
        self.classifier_type = classifier_type
        
        # Initialiser le classifieur
        if classifier_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
        elif classifier_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1, 
                random_state=42
            )
        elif classifier_type == "logistic":
            self.model = LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42
            )
        elif classifier_type == "svm":
            self.model = SVC(
                C=1.0, 
                kernel='rbf', 
                probability=True, 
                random_state=42
            )
        elif classifier_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=1000, 
                random_state=42
            )
        elif classifier_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1, 
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def train(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005, 
              test_size: float = 0.2, include_indicators: bool = True) -> Dict[str, float]:
        """
        Entraîne le classifieur sur les données.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            horizon: Horizon de prédiction en périodes
            threshold: Seuil pour la classification (en pourcentage)
            test_size: Proportion des données à utiliser pour le test
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Dictionnaire des métriques d'évaluation sur l'ensemble de test
        """
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        # Créer la variable cible
        target = self.feature_engineering.create_target(df, horizon, threshold)
        
        # Aligner les index
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Supprimer les lignes avec des valeurs NaN
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        if len(features) == 0:
            logger.warning("No valid data for training")
            return {}
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features, 'standard', True)
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=test_size, shuffle=False
        )
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Évaluer sur l'ensemble de test
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Trained {self.name} with {len(X_train)} samples, test metrics: {metrics}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, include_indicators: bool = True) -> pd.Series:
        """
        Fait des prédictions avec le classifieur.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Series avec les prédictions de classe (-1, 0, 1)
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained")
            return pd.Series(0, index=df.index)
        
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        if features.empty:
            return pd.Series(0, index=df.index)
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features, 'standard', False)
        
        # Faire les prédictions
        predictions = self.model.predict(scaled_features)
        
        # Créer une Series avec les prédictions
        pred_series = pd.Series(predictions, index=scaled_features.index)
        
        return pred_series
    
    def predict_proba(self, df: pd.DataFrame, include_indicators: bool = True) -> pd.DataFrame:
        """
        Fait des prédictions de probabilité avec le classifieur.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            DataFrame avec les probabilités pour chaque classe
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained")
            return pd.DataFrame(index=df.index)
        
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        if features.empty:
            return pd.DataFrame(index=df.index)
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features, 'standard', False)
        
        # Faire les prédictions de probabilité
        proba = self.model.predict_proba(scaled_features)
        
        # Créer un DataFrame avec les probabilités
        classes = self.model.classes_
        proba_df = pd.DataFrame(
            proba, 
            index=scaled_features.index, 
            columns=[f"class_{c}" for c in classes]
        )
        
        return proba_df
    
    def evaluate(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005, 
                include_indicators: bool = True) -> Dict[str, float]:
        """
        Évalue les performances du classifieur.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            horizon: Horizon de prédiction en périodes
            threshold: Seuil pour la classification (en pourcentage)
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained")
            return {}
        
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        # Créer la variable cible
        target = self.feature_engineering.create_target(df, horizon, threshold)
        
        # Aligner les index
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Supprimer les lignes avec des valeurs NaN
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        if len(features) == 0:
            logger.warning("No valid data for evaluation")
            return {}
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features, 'standard', False)
        
        # Faire les prédictions
        predictions = self.model.predict(scaled_features)
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(target, predictions),
            'precision': precision_score(target, predictions, average='weighted'),
            'recall': recall_score(target, predictions, average='weighted'),
            'f1': f1_score(target, predictions, average='weighted')
        }
        
        return metrics


class MarketRegimeClusterer(BaseModel):
    """
    Modèle de clustering pour identifier les régimes de marché.
    """
    
    def __init__(self, name: str = "market_regime_clusterer", n_clusters: int = 3):
        """
        Initialise un clusterer de régimes de marché.
        
        Args:
            name: Nom du modèle
            n_clusters: Nombre de clusters (régimes)
        """
        super().__init__(name, ModelType.CLUSTERING)
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.regime_features = [
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'ma_5d', 'ma_10d', 'ma_20d',
            'std_5d', 'std_10d', 'std_20d',
            'volume_ma_ratio_5d', 'volume_ma_ratio_10d'
        ]
    
    def train(self, df: pd.DataFrame, include_indicators: bool = False) -> None:
        """
        Entraîne le clusterer sur les données.
        
        Args:
            df: DataFrame contenant les données OHLCV
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
        """
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        # Sélectionner les caractéristiques pertinentes pour le clustering
        regime_features = [f for f in self.regime_features if f in features.columns]
        
        if not regime_features:
            logger.warning("No valid regime features found")
            return
        
        features_subset = features[regime_features]
        
        # Supprimer les lignes avec des valeurs NaN
        features_subset = features_subset.dropna()
        
        if len(features_subset) == 0:
            logger.warning("No valid data for training")
            return
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features_subset, 'standard', True)
        
        # Entraîner le modèle
        self.model.fit(scaled_features)
        self.trained = True
        
        logger.info(f"Trained {self.name} with {len(scaled_features)} samples")
    
    def predict(self, df: pd.DataFrame, include_indicators: bool = False) -> pd.Series:
        """
        Identifie les régimes de marché.
        
        Args:
            df: DataFrame contenant les données OHLCV
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Series avec les identifiants de régime
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained")
            return pd.Series(0, index=df.index)
        
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        # Sélectionner les caractéristiques pertinentes pour le clustering
        regime_features = [f for f in self.regime_features if f in features.columns]
        
        if not regime_features:
            logger.warning("No valid regime features found")
            return pd.Series(0, index=df.index)
        
        features_subset = features[regime_features]
        
        # Supprimer les lignes avec des valeurs NaN
        valid_indices = features_subset.dropna().index
        features_subset = features_subset.loc[valid_indices]
        
        if len(features_subset) == 0:
            return pd.Series(0, index=df.index)
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features_subset, 'standard', False)
        
        # Faire les prédictions
        regimes = self.model.predict(scaled_features)
        
        # Créer une Series avec les régimes
        regime_series = pd.Series(regimes, index=scaled_features.index)
        
        return regime_series
    
    def evaluate(self, df: pd.DataFrame, include_indicators: bool = False) -> Dict[str, float]:
        """
        Évalue la qualité du clustering.
        
        Args:
            df: DataFrame contenant les données OHLCV
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        if not self.trained:
            logger.warning(f"Model {self.name} is not trained")
            return {}
        
        # Créer les caractéristiques
        features = self.feature_engineering.create_features(df, include_indicators)
        
        # Sélectionner les caractéristiques pertinentes pour le clustering
        regime_features = [f for f in self.regime_features if f in features.columns]
        
        if not regime_features:
            logger.warning("No valid regime features found")
            return {}
        
        features_subset = features[regime_features]
        
        # Supprimer les lignes avec des valeurs NaN
        features_subset = features_subset.dropna()
        
        if len(features_subset) == 0:
            logger.warning("No valid data for evaluation")
            return {}
        
        # Normaliser les caractéristiques
        scaled_features = self.feature_engineering.scale_features(features_subset, 'standard', False)
        
        # Calculer l'inertie (somme des distances au carré)
        inertia = self.model.inertia_
        
        # Calculer la silhouette moyenne (mesure de la qualité du clustering)
        from sklearn.metrics import silhouette_score
        labels = self.model.predict(scaled_features)
        silhouette = silhouette_score(scaled_features, labels)
        
        metrics = {
            'inertia': inertia,
            'silhouette': silhouette
        }
        
        return metrics


class ModelEnsemble:
    """
    Ensemble de modèles pour combiner les prédictions.
    """
    
    def __init__(self, name: str = "model_ensemble"):
        """
        Initialise un ensemble de modèles.
        
        Args:
            name: Nom de l'ensemble
        """
        self.name = name
        self.models = {}
        self.weights = {}
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Ajoute un modèle à l'ensemble.
        
        Args:
            model: Instance de BaseModel
            weight: Poids du modèle dans l'ensemble
        """
        self.models[model.name] = model
        self.weights[model.name] = weight
    
    def predict(self, df: pd.DataFrame, include_indicators: bool = True) -> pd.Series:
        """
        Fait des prédictions combinées avec l'ensemble de modèles.
        
        Args:
            df: DataFrame contenant les données OHLCV et éventuellement des indicateurs
            include_indicators: Si True, inclut les indicateurs techniques comme caractéristiques
            
        Returns:
            Series avec les prédictions combinées
        """
        if not self.models:
            logger.warning("No models in ensemble")
            return pd.Series(0, index=df.index)
        
        # Faire les prédictions avec chaque modèle
        predictions = {}
        for name, model in self.models.items():
            if model.trained:
                pred = model.predict(df, include_indicators)
                predictions[name] = pred
        
        if not predictions:
            logger.warning("No trained models in ensemble")
            return pd.Series(0, index=df.index)
        
        # Combiner les prédictions en fonction des poids
        combined = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            # Aligner les index
            aligned_pred = pred.reindex(combined.index, fill_value=0)
            combined += aligned_pred * weight
            total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        # Discrétiser les prédictions combinées
        discretized = pd.Series(0, index=combined.index)
        discretized[combined > 0.2] = 1
        discretized[combined < -0.2] = -1
        
        return discretized
    
    def save(self, directory: str) -> None:
        """
        Sauvegarde tous les modèles de l'ensemble.
        
        Args:
            directory: Chemin du répertoire où sauvegarder les modèles
        """
        os.makedirs(directory, exist_ok=True)
        
        # Sauvegarder les poids
        weights_path = os.path.join(directory, f"{self.name}_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f)
        
        # Sauvegarder chaque modèle
        for name, model in self.models.items():
            model_dir = os.path.join(directory, name)
            model.save(model_dir)
        
        logger.info(f"Saved ensemble {self.name} to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Charge tous les modèles de l'ensemble.
        
        Args:
            directory: Chemin du répertoire contenant les modèles
        """
        # Charger les poids
        weights_path = os.path.join(directory, f"{self.name}_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.weights = json.load(f)
        
        # Charger chaque modèle
        for name, model in self.models.items():
            model_dir = os.path.join(directory, name)
            if os.path.exists(model_dir):
                model.load(model_dir)
        
        logger.info(f"Loaded ensemble {self.name} from {directory}")


def create_default_model_ensemble() -> ModelEnsemble:
    """
    Crée un ensemble de modèles par défaut.
    
    Returns:
        ModelEnsemble: Ensemble de modèles par défaut
    """
    ensemble = ModelEnsemble("default_ensemble")
    
    # Ajouter des classifieurs de direction avec différents algorithmes
    ensemble.add_model(DirectionClassifier("rf_classifier", "random_forest"), weight=1.0)
    ensemble.add_model(DirectionClassifier("gb_classifier", "gradient_boosting"), weight=1.0)
    ensemble.add_model(DirectionClassifier("xgb_classifier", "xgboost"), weight=1.5)
    
    # Ajouter un clusterer de régimes de marché
    ensemble.add_model(MarketRegimeClusterer("regime_clusterer", n_clusters=3), weight=0.5)
    
    return ensemble


if __name__ == "__main__":
    # Exemple d'utilisation
    import yfinance as yf
    
    # Télécharger des données historiques
    data = yf.download("BTC-USD", period="1y", interval="1d")
    
    # Créer un classifieur de direction
    classifier = DirectionClassifier("example_classifier", "random_forest")
    
    # Entraîner le classifieur
    metrics = classifier.train(data, horizon=1, threshold=0.01, test_size=0.2)
    print(f"Training metrics: {metrics}")
    
    # Faire des prédictions
    predictions = classifier.predict(data.tail(10))
    print("\nPredictions:")
    print(predictions)
    
    # Créer un clusterer de régimes de marché
    clusterer = MarketRegimeClusterer("example_clusterer", n_clusters=3)
    
    # Entraîner le clusterer
    clusterer.train(data)
    
    # Identifier les régimes
    regimes = clusterer.predict(data.tail(10))
    print("\nMarket Regimes:")
    print(regimes)
    
    # Créer un ensemble de modèles
    ensemble = create_default_model_ensemble()
    
    # Ajouter les modèles entraînés
    ensemble.models["rf_classifier"] = classifier
    ensemble.models["regime_clusterer"] = clusterer
    
    # Faire des prédictions combinées
    combined_predictions = ensemble.predict(data.tail(10))
    print("\nCombined Predictions:")
    print(combined_predictions)

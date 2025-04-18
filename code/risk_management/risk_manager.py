#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de gestion du risque adaptatif pour le système de trading automatisé Bitget.

Ce module implémente un système de gestion du risque qui s'adapte dynamiquement
aux conditions du marché et aux performances du trading.
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

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('risk_management')


class RiskLevel(Enum):
    """Niveaux de risque pour le système de trading."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PositionSizing:
    """
    Classe pour le dimensionnement des positions en fonction du risque.
    """
    
    def __init__(self, 
                 base_risk_per_trade: float = 0.01, 
                 max_risk_per_trade: float = 0.03,
                 max_portfolio_risk: float = 0.10,
                 max_correlated_risk: float = 0.15,
                 min_position_size: float = 0.01,
                 max_position_size: float = 0.20):
        """
        Initialise le gestionnaire de dimensionnement des positions.
        
        Args:
            base_risk_per_trade: Risque de base par trade (en % du capital)
            max_risk_per_trade: Risque maximum par trade (en % du capital)
            max_portfolio_risk: Risque maximum pour l'ensemble du portefeuille (en % du capital)
            max_correlated_risk: Risque maximum pour des actifs corrélés (en % du capital)
            min_position_size: Taille minimale d'une position (en % du capital)
            max_position_size: Taille maximale d'une position (en % du capital)
        """
        self.base_risk_per_trade = base_risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_risk = max_correlated_risk
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, 
                               capital: float, 
                               entry_price: float, 
                               stop_loss: float, 
                               risk_level: RiskLevel = RiskLevel.MEDIUM,
                               volatility_factor: float = 1.0,
                               signal_strength: float = 0.5,
                               current_portfolio_risk: float = 0.0) -> Dict[str, float]:
        """
        Calcule la taille optimale d'une position.
        
        Args:
            capital: Capital total disponible
            entry_price: Prix d'entrée prévu
            stop_loss: Prix du stop loss
            risk_level: Niveau de risque global
            volatility_factor: Facteur de volatilité (1.0 = volatilité normale)
            signal_strength: Force du signal de trading (0.0 à 1.0)
            current_portfolio_risk: Risque actuel du portefeuille (en % du capital)
            
        Returns:
            Dictionnaire contenant les informations de dimensionnement de la position
        """
        # Calculer le risque par trade en fonction du niveau de risque
        risk_multipliers = {
            RiskLevel.VERY_LOW: 0.5,
            RiskLevel.LOW: 0.75,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 1.25,
            RiskLevel.VERY_HIGH: 1.5
        }
        
        risk_per_trade = self.base_risk_per_trade * risk_multipliers[risk_level]
        
        # Ajuster le risque en fonction de la force du signal
        risk_per_trade = risk_per_trade * (0.5 + signal_strength * 0.5)
        
        # Ajuster le risque en fonction de la volatilité
        if volatility_factor > 1.5:
            risk_per_trade = risk_per_trade / (volatility_factor * 0.5)
        
        # Limiter le risque par trade
        risk_per_trade = min(risk_per_trade, self.max_risk_per_trade)
        
        # Vérifier si le risque total du portefeuille serait dépassé
        remaining_risk = self.max_portfolio_risk - current_portfolio_risk
        if remaining_risk < risk_per_trade:
            risk_per_trade = max(0, remaining_risk)
        
        # Calculer le risque en valeur absolue
        risk_amount = capital * risk_per_trade
        
        # Calculer la distance au stop loss en pourcentage
        if entry_price > stop_loss:  # Position longue
            stop_distance_pct = (entry_price - stop_loss) / entry_price
        else:  # Position courte
            stop_distance_pct = (stop_loss - entry_price) / entry_price
        
        # Calculer la taille de la position en unités de l'actif
        if stop_distance_pct > 0:
            position_value = risk_amount / stop_distance_pct
        else:
            position_value = 0
        
        # Limiter la taille de la position en pourcentage du capital
        position_value = min(position_value, capital * self.max_position_size)
        position_value = max(position_value, capital * self.min_position_size)
        
        # Calculer le nombre d'unités
        units = position_value / entry_price
        
        # Calculer le levier effectif
        leverage = position_value / (position_value * stop_distance_pct)
        
        return {
            "position_value": position_value,
            "units": units,
            "risk_amount": risk_amount,
            "risk_percentage": risk_per_trade * 100,
            "stop_distance_pct": stop_distance_pct * 100,
            "leverage": leverage
        }


class StopLossManager:
    """
    Classe pour la gestion des stop loss adaptatifs.
    """
    
    def __init__(self, 
                 atr_multiplier: float = 2.0,
                 min_stop_distance: float = 0.01,
                 max_stop_distance: float = 0.10,
                 trailing_activation: float = 0.02,
                 trailing_step: float = 0.005):
        """
        Initialise le gestionnaire de stop loss.
        
        Args:
            atr_multiplier: Multiplicateur pour l'ATR (Average True Range)
            min_stop_distance: Distance minimale du stop loss (en % du prix)
            max_stop_distance: Distance maximale du stop loss (en % du prix)
            trailing_activation: Profit nécessaire pour activer le trailing stop (en % du prix)
            trailing_step: Pas du trailing stop (en % du prix)
        """
        self.atr_multiplier = atr_multiplier
        self.min_stop_distance = min_stop_distance
        self.max_stop_distance = max_stop_distance
        self.trailing_activation = trailing_activation
        self.trailing_step = trailing_step
    
    def calculate_initial_stop(self, 
                              entry_price: float, 
                              atr: float, 
                              position_type: str,
                              support_resistance: Optional[float] = None,
                              volatility_factor: float = 1.0) -> float:
        """
        Calcule le niveau de stop loss initial.
        
        Args:
            entry_price: Prix d'entrée de la position
            atr: Valeur de l'ATR (Average True Range)
            position_type: Type de position ('long' ou 'short')
            support_resistance: Niveau de support/résistance proche (optionnel)
            volatility_factor: Facteur de volatilité (1.0 = volatilité normale)
            
        Returns:
            Prix du stop loss
        """
        # Calculer la distance du stop en fonction de l'ATR
        stop_distance = atr * self.atr_multiplier * volatility_factor
        
        # Limiter la distance en pourcentage du prix
        stop_distance_pct = stop_distance / entry_price
        stop_distance_pct = max(stop_distance_pct, self.min_stop_distance)
        stop_distance_pct = min(stop_distance_pct, self.max_stop_distance)
        
        # Recalculer la distance en valeur absolue
        stop_distance = entry_price * stop_distance_pct
        
        # Calculer le niveau de stop loss
        if position_type.lower() == 'long':
            stop_level = entry_price - stop_distance
            
            # Ajuster en fonction du support si disponible
            if support_resistance is not None and support_resistance < entry_price:
                stop_level = max(stop_level, support_resistance * 0.99)
        
        elif position_type.lower() == 'short':
            stop_level = entry_price + stop_distance
            
            # Ajuster en fonction de la résistance si disponible
            if support_resistance is not None and support_resistance > entry_price:
                stop_level = min(stop_level, support_resistance * 1.01)
        
        else:
            raise ValueError(f"Invalid position type: {position_type}")
        
        return stop_level
    
    def update_trailing_stop(self, 
                            current_price: float, 
                            entry_price: float,
                            current_stop: float,
                            position_type: str,
                            highest_lowest_price: Optional[float] = None) -> float:
        """
        Met à jour le niveau de trailing stop.
        
        Args:
            current_price: Prix actuel
            entry_price: Prix d'entrée de la position
            current_stop: Niveau de stop loss actuel
            position_type: Type de position ('long' ou 'short')
            highest_lowest_price: Prix le plus haut/bas atteint depuis l'entrée
            
        Returns:
            Nouveau niveau de stop loss
        """
        # Si highest_lowest_price n'est pas fourni, utiliser le prix actuel
        if highest_lowest_price is None:
            highest_lowest_price = current_price
        
        # Calculer le profit actuel en pourcentage
        if position_type.lower() == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            
            # Vérifier si le trailing stop doit être activé
            if profit_pct >= self.trailing_activation:
                # Calculer le nouveau niveau de stop
                new_stop = highest_lowest_price * (1 - self.trailing_step)
                
                # Ne mettre à jour que si le nouveau stop est plus élevé
                if new_stop > current_stop:
                    return new_stop
        
        elif position_type.lower() == 'short':
            profit_pct = (entry_price - current_price) / entry_price
            
            # Vérifier si le trailing stop doit être activé
            if profit_pct >= self.trailing_activation:
                # Calculer le nouveau niveau de stop
                new_stop = highest_lowest_price * (1 + self.trailing_step)
                
                # Ne mettre à jour que si le nouveau stop est plus bas
                if new_stop < current_stop:
                    return new_stop
        
        else:
            raise ValueError(f"Invalid position type: {position_type}")
        
        # Si aucune mise à jour n'est nécessaire, retourner le stop actuel
        return current_stop


class TakeProfitManager:
    """
    Classe pour la gestion des prises de profit.
    """
    
    def __init__(self, 
                 risk_reward_ratios: List[float] = [1.5, 2.5, 4.0],
                 profit_distribution: List[float] = [0.3, 0.3, 0.4],
                 min_first_target: float = 0.01,
                 adjust_for_volatility: bool = True):
        """
        Initialise le gestionnaire de prises de profit.
        
        Args:
            risk_reward_ratios: Ratios risque/récompense pour les cibles de profit
            profit_distribution: Distribution du profit entre les cibles (doit sommer à 1.0)
            min_first_target: Profit minimum pour la première cible (en % du prix)
            adjust_for_volatility: Si True, ajuste les cibles en fonction de la volatilité
        """
        self.risk_reward_ratios = risk_reward_ratios
        self.profit_distribution = profit_distribution
        self.min_first_target = min_first_target
        self.adjust_for_volatility = adjust_for_volatility
        
        # Vérifier que la distribution somme à 1.0
        if abs(sum(profit_distribution) - 1.0) > 0.001:
            raise ValueError("Profit distribution must sum to 1.0")
        
        # Vérifier que les listes ont la même longueur
        if len(risk_reward_ratios) != len(profit_distribution):
            raise ValueError("Risk/reward ratios and profit distribution must have the same length")
    
    def calculate_take_profit_levels(self, 
                                    entry_price: float, 
                                    stop_loss: float, 
                                    position_type: str,
                                    volatility_factor: float = 1.0) -> Dict[str, Any]:
        """
        Calcule les niveaux de prise de profit.
        
        Args:
            entry_price: Prix d'entrée de la position
            stop_loss: Niveau de stop loss
            position_type: Type de position ('long' ou 'short')
            volatility_factor: Facteur de volatilité (1.0 = volatilité normale)
            
        Returns:
            Dictionnaire contenant les niveaux de prise de profit et la distribution
        """
        # Calculer la distance au stop loss
        if position_type.lower() == 'long':
            stop_distance = entry_price - stop_loss
            if stop_distance <= 0:
                raise ValueError("Stop loss must be below entry price for long positions")
        
        elif position_type.lower() == 'short':
            stop_distance = stop_loss - entry_price
            if stop_distance <= 0:
                raise ValueError("Stop loss must be above entry price for short positions")
        
        else:
            raise ValueError(f"Invalid position type: {position_type}")
        
        # Ajuster les ratios en fonction de la volatilité si nécessaire
        adjusted_ratios = self.risk_reward_ratios.copy()
        if self.adjust_for_volatility:
            if volatility_factor > 1.0:
                # Augmenter les ratios pour les marchés plus volatils
                adjusted_ratios = [r * (1 + (volatility_factor - 1) * 0.5) for r in adjusted_ratios]
            elif volatility_factor < 1.0:
                # Diminuer les ratios pour les marchés moins volatils
                adjusted_ratios = [r * (1 - (1 - volatility_factor) * 0.5) for r in adjusted_ratios]
        
        # Calculer les niveaux de prise de profit
        take_profit_levels = []
        for ratio in adjusted_ratios:
            if position_type.lower() == 'long':
                tp_level = entry_price + (stop_distance * ratio)
            else:
                tp_level = entry_price - (stop_distance * ratio)
            
            take_profit_levels.append(tp_level)
        
        # Vérifier que la première cible respecte le profit minimum
        min_first_target_price = entry_price * (1 + self.min_first_target) if position_type.lower() == 'long' else entry_price * (1 - self.min_first_target)
        
        if position_type.lower() == 'long' and take_profit_levels[0] < min_first_target_price:
            take_profit_levels[0] = min_first_target_price
        elif position_type.lower() == 'short' and take_profit_levels[0] > min_first_target_price:
            take_profit_levels[0] = min_first_target_price
        
        return {
            "levels": take_profit_levels,
            "distribution": self.profit_distribution
        }


class LeverageManager:
    """
    Classe pour la gestion adaptative de l'effet de levier.
    """
    
    def __init__(self, 
                 base_leverage: float = 3.0,
                 min_leverage: float = 1.0,
                 max_leverage: float = 10.0,
                 volatility_adjustment: bool = True,
                 performance_adjustment: bool = True):
        """
        Initialise le gestionnaire d'effet de levier.
        
        Args:
            base_leverage: Effet de levier de base
            min_leverage: Effet de levier minimum
            max_leverage: Effet de levier maximum
            volatility_adjustment: Si True, ajuste le levier en fonction de la volatilité
            performance_adjustment: Si True, ajuste le levier en fonction des performances
        """
        self.base_leverage = base_leverage
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.volatility_adjustment = volatility_adjustment
        self.performance_adjustment = performance_adjustment
    
    def calculate_optimal_leverage(self, 
                                  volatility_factor: float = 1.0,
                                  win_rate: float = 0.5,
                                  recent_performance: float = 0.0,
                                  market_regime: str = "normal",
                                  signal_strength: float = 0.5) -> float:
        """
        Calcule l'effet de levier optimal.
        
        Args:
            volatility_factor: Facteur de volatilité (1.0 = volatilité normale)
            win_rate: Taux de réussite historique (0.0 à 1.0)
            recent_performance: Performance récente (-1.0 à 1.0)
            market_regime: Régime de marché ('low_volatility', 'normal', 'high_volatility', 'trending', 'ranging')
            signal_strength: Force du signal de trading (0.0 à 1.0)
            
        Returns:
            Effet de levier optimal
        """
        # Commencer avec le levier de base
        leverage = self.base_leverage
        
        # Ajuster en fonction de la volatilité
        if self.volatility_adjustment:
            if volatility_factor > 1.0:
                # Réduire le levier pour les marchés plus volatils
                leverage = leverage / (volatility_factor ** 0.5)
            elif volatility_factor < 1.0:
                # Augmenter le levier pour les marchés moins volatils
                leverage = leverage * (1 / volatility_factor) ** 0.25
        
        # Ajuster en fonction des performances
        if self.performance_adjustment:
            # Ajuster en fonction du taux de réussite
            if win_rate > 0.5:
                leverage = leverage * (1 + (win_rate - 0.5) * 0.5)
            elif win_rate < 0.5:
                leverage = leverage * (1 - (0.5 - win_rate) * 1.0)
            
            # Ajuster en fonction de la performance récente
            if recent_performance > 0:
                leverage = leverage * (1 + recent_performance * 0.2)
            elif recent_performance < 0:
                leverage = leverage * (1 + recent_performance * 0.5)
        
        # Ajuster en fonction du régime de marché
        market_regime_multipliers = {
            "low_volatility": 1.2,
            "normal": 1.0,
            "high_volatility": 0.7,
            "trending": 1.1,
            "ranging": 0.9
        }
        
        if market_regime in market_regime_multipliers:
            leverage = leverage * market_regime_multipliers[market_regime]
        
        # Ajuster en fonction de la force du signal
        leverage = leverage * (0.8 + signal_strength * 0.4)
        
        # Limiter le levier aux bornes définies
        leverage = max(self.min_leverage, min(leverage, self.max_leverage))
        
        return leverage


class PortfolioRiskManager:
    """
    Classe pour la gestion du risque au niveau du portefeuille.
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.10,
                 max_correlated_risk: float = 0.15,
                 max_drawdown_threshold: float = 0.15,
                 risk_reduction_factor: float = 0.5,
                 correlation_threshold: float = 0.7):
        """
        Initialise le gestionnaire de risque du portefeuille.
        
        Args:
            max_portfolio_risk: Risque maximum pour l'ensemble du portefeuille (en % du capital)
            max_correlated_risk: Risque maximum pour des actifs corrélés (en % du capital)
            max_drawdown_threshold: Seuil de drawdown maximum avant réduction du risque
            risk_reduction_factor: Facteur de réduction du risque en cas de dépassement du drawdown
            correlation_threshold: Seuil de corrélation pour considérer des actifs comme corrélés
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_risk = max_correlated_risk
        self.max_drawdown_threshold = max_drawdown_threshold
        self.risk_reduction_factor = risk_reduction_factor
        self.correlation_threshold = correlation_threshold
        
        # Historique des performances
        self.equity_history = []
        self.drawdown_history = []
        self.peak_equity = 0.0
    
    def update_equity(self, current_equity: float) -> Dict[str, float]:
        """
        Met à jour l'historique des performances et calcule le drawdown.
        
        Args:
            current_equity: Valeur actuelle du portefeuille
            
        Returns:
            Dictionnaire contenant les informations de drawdown
        """
        self.equity_history.append(current_equity)
        
        # Mettre à jour le pic d'equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculer le drawdown actuel
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            current_drawdown = 0.0
        
        self.drawdown_history.append(current_drawdown)
        
        return {
            "current_equity": current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": current_drawdown,
            "max_drawdown": max(self.drawdown_history) if self.drawdown_history else 0.0
        }
    
    def calculate_portfolio_risk_adjustment(self) -> float:
        """
        Calcule un facteur d'ajustement du risque en fonction du drawdown.
        
        Returns:
            Facteur d'ajustement du risque (1.0 = pas d'ajustement)
        """
        if not self.drawdown_history:
            return 1.0
        
        current_drawdown = self.drawdown_history[-1]
        
        # Si le drawdown dépasse le seuil, réduire le risque
        if current_drawdown > self.max_drawdown_threshold:
            # Réduction proportionnelle au dépassement
            excess_drawdown = current_drawdown - self.max_drawdown_threshold
            adjustment_factor = 1.0 - (excess_drawdown / self.max_drawdown_threshold) * self.risk_reduction_factor
            
            # Limiter la réduction à un minimum de 10% du risque normal
            adjustment_factor = max(0.1, adjustment_factor)
            
            return adjustment_factor
        
        return 1.0
    
    def check_correlation_risk(self, 
                              positions: List[Dict[str, Any]], 
                              correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Vérifie le risque lié aux corrélations entre les positions.
        
        Args:
            positions: Liste des positions actuelles
            correlation_matrix: Matrice de corrélation entre les actifs
            
        Returns:
            Dictionnaire contenant les informations de risque de corrélation
        """
        if not positions:
            return {"correlated_groups": [], "risk_exceeded": False}
        
        # Extraire les symboles des positions
        symbols = [p["symbol"] for p in positions]
        
        # Vérifier que tous les symboles sont dans la matrice de corrélation
        valid_symbols = [s for s in symbols if s in correlation_matrix.index and s in correlation_matrix.columns]
        
        if not valid_symbols:
            return {"correlated_groups": [], "risk_exceeded": False}
        
        # Extraire la sous-matrice de corrélation pour les symboles actuels
        sub_matrix = correlation_matrix.loc[valid_symbols, valid_symbols]
        
        # Identifier les groupes d'actifs corrélés
        correlated_groups = []
        processed_symbols = set()
        
        for symbol in valid_symbols:
            if symbol in processed_symbols:
                continue
            
            # Trouver tous les symboles corrélés à celui-ci
            correlated = sub_matrix[symbol][sub_matrix[symbol].abs() > self.correlation_threshold].index.tolist()
            
            if len(correlated) > 1:  # Au moins un autre symbole corrélé
                correlated_groups.append(correlated)
                processed_symbols.update(correlated)
        
        # Calculer le risque pour chaque groupe corrélé
        group_risks = []
        risk_exceeded = False
        
        for group in correlated_groups:
            # Trouver les positions correspondantes
            group_positions = [p for p in positions if p["symbol"] in group]
            
            # Calculer le risque total du groupe
            group_risk = sum(p.get("risk_amount", 0) for p in group_positions)
            group_risk_pct = group_risk / sum(p.get("position_value", 0) for p in positions) if positions else 0
            
            group_risks.append({
                "symbols": group,
                "risk_amount": group_risk,
                "risk_percentage": group_risk_pct * 100
            })
            
            # Vérifier si le risque dépasse le maximum autorisé
            if group_risk_pct > self.max_correlated_risk:
                risk_exceeded = True
        
        return {
            "correlated_groups": group_risks,
            "risk_exceeded": risk_exceeded
        }
    
    def calculate_max_positions(self, 
                               current_drawdown: float, 
                               volatility_factor: float = 1.0,
                               win_rate: float = 0.5) -> int:
        """
        Calcule le nombre maximum de positions simultanées en fonction des conditions.
        
        Args:
            current_drawdown: Drawdown actuel
            volatility_factor: Facteur de volatilité du marché
            win_rate: Taux de réussite historique
            
        Returns:
            Nombre maximum de positions recommandé
        """
        # Base: entre 3 et 10 positions selon le taux de réussite
        base_max = 3 + int(win_rate * 7)
        
        # Ajuster en fonction du drawdown
        if current_drawdown > self.max_drawdown_threshold:
            drawdown_factor = 1.0 - (current_drawdown - self.max_drawdown_threshold) / (1.0 - self.max_drawdown_threshold)
            drawdown_factor = max(0.3, drawdown_factor)
            base_max = int(base_max * drawdown_factor)
        
        # Ajuster en fonction de la volatilité
        if volatility_factor > 1.5:
            base_max = int(base_max * 0.7)
        elif volatility_factor < 0.7:
            base_max = int(base_max * 1.2)
        
        # Limiter entre 1 et 15 positions
        return max(1, min(15, base_max))


class RiskManager:
    """
    Classe principale pour la gestion du risque adaptatif.
    """
    
    def __init__(self, 
                 initial_capital: float,
                 risk_level: RiskLevel = RiskLevel.MEDIUM,
                 auto_adjust_risk: bool = True):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            initial_capital: Capital initial
            risk_level: Niveau de risque global
            auto_adjust_risk: Si True, ajuste automatiquement le niveau de risque
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_level = risk_level
        self.auto_adjust_risk = auto_adjust_risk
        
        # Initialiser les composants
        self.position_sizing = PositionSizing()
        self.stop_loss_manager = StopLossManager()
        self.take_profit_manager = TakeProfitManager()
        self.leverage_manager = LeverageManager()
        self.portfolio_risk_manager = PortfolioRiskManager()
        
        # Statistiques de performance
        self.trades_history = []
        self.win_rate = 0.5
        self.avg_win_loss_ratio = 1.0
        self.recent_performance = 0.0
    
    def update_capital(self, new_capital: float) -> None:
        """
        Met à jour le capital actuel et les statistiques de drawdown.
        
        Args:
            new_capital: Nouveau montant du capital
        """
        # Mettre à jour le capital
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        # Mettre à jour les statistiques de drawdown
        drawdown_info = self.portfolio_risk_manager.update_equity(new_capital)
        
        # Calculer la performance récente
        if old_capital > 0:
            self.recent_performance = (new_capital - old_capital) / old_capital
        
        # Ajuster automatiquement le niveau de risque si nécessaire
        if self.auto_adjust_risk:
            self._adjust_risk_level(drawdown_info["current_drawdown"])
    
    def _adjust_risk_level(self, current_drawdown: float) -> None:
        """
        Ajuste automatiquement le niveau de risque en fonction du drawdown.
        
        Args:
            current_drawdown: Drawdown actuel
        """
        if current_drawdown > 0.25:
            self.risk_level = RiskLevel.VERY_LOW
        elif current_drawdown > 0.15:
            self.risk_level = RiskLevel.LOW
        elif current_drawdown > 0.10:
            self.risk_level = RiskLevel.MEDIUM
        elif self.win_rate > 0.6 and self.avg_win_loss_ratio > 1.2:
            if self.recent_performance > 0.05:
                self.risk_level = RiskLevel.HIGH
            elif self.recent_performance > 0.10:
                self.risk_level = RiskLevel.VERY_HIGH
            else:
                self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.MEDIUM
    
    def update_performance_stats(self, trade_result: Dict[str, Any]) -> None:
        """
        Met à jour les statistiques de performance après un trade.
        
        Args:
            trade_result: Résultat du trade
        """
        self.trades_history.append(trade_result)
        
        # Limiter l'historique aux 100 derniers trades
        if len(self.trades_history) > 100:
            self.trades_history = self.trades_history[-100:]
        
        # Calculer le taux de réussite
        wins = sum(1 for t in self.trades_history if t.get("profit", 0) > 0)
        if self.trades_history:
            self.win_rate = wins / len(self.trades_history)
        
        # Calculer le ratio gain/perte moyen
        wins = [t.get("profit", 0) for t in self.trades_history if t.get("profit", 0) > 0]
        losses = [abs(t.get("profit", 0)) for t in self.trades_history if t.get("profit", 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 1
        
        self.avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Calculer la performance récente (10 derniers trades)
        recent_trades = self.trades_history[-10:] if len(self.trades_history) >= 10 else self.trades_history
        recent_profit = sum(t.get("profit", 0) for t in recent_trades)
        
        if recent_trades:
            recent_capital = sum(t.get("capital", 0) for t in recent_trades) / len(recent_trades)
            if recent_capital > 0:
                self.recent_performance = recent_profit / recent_capital
    
    def calculate_trade_parameters(self, 
                                  symbol: str,
                                  entry_price: float,
                                  position_type: str,
                                  atr: float,
                                  volatility_factor: float = 1.0,
                                  signal_strength: float = 0.5,
                                  support_resistance: Optional[float] = None,
                                  market_regime: str = "normal",
                                  current_positions: List[Dict[str, Any]] = None,
                                  correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calcule tous les paramètres pour un trade.
        
        Args:
            symbol: Symbole de l'actif
            entry_price: Prix d'entrée prévu
            position_type: Type de position ('long' ou 'short')
            atr: Valeur de l'ATR (Average True Range)
            volatility_factor: Facteur de volatilité (1.0 = volatilité normale)
            signal_strength: Force du signal de trading (0.0 à 1.0)
            support_resistance: Niveau de support/résistance proche (optionnel)
            market_regime: Régime de marché
            current_positions: Liste des positions actuelles
            correlation_matrix: Matrice de corrélation entre les actifs
            
        Returns:
            Dictionnaire contenant tous les paramètres du trade
        """
        if current_positions is None:
            current_positions = []
        
        # Calculer le risque actuel du portefeuille
        current_portfolio_risk = sum(p.get("risk_percentage", 0) / 100 for p in current_positions)
        
        # Calculer le facteur d'ajustement du risque en fonction du drawdown
        risk_adjustment = self.portfolio_risk_manager.calculate_portfolio_risk_adjustment()
        
        # Calculer le stop loss
        stop_loss = self.stop_loss_manager.calculate_initial_stop(
            entry_price, atr, position_type, support_resistance, volatility_factor
        )
        
        # Calculer l'effet de levier optimal
        leverage = self.leverage_manager.calculate_optimal_leverage(
            volatility_factor, self.win_rate, self.recent_performance, market_regime, signal_strength
        )
        
        # Calculer la taille de la position
        position_info = self.position_sizing.calculate_position_size(
            self.current_capital,
            entry_price,
            stop_loss,
            self.risk_level,
            volatility_factor,
            signal_strength,
            current_portfolio_risk
        )
        
        # Appliquer l'ajustement du risque
        position_info["risk_amount"] *= risk_adjustment
        position_info["risk_percentage"] *= risk_adjustment
        position_info["position_value"] *= risk_adjustment
        position_info["units"] *= risk_adjustment
        
        # Calculer les niveaux de prise de profit
        take_profit_info = self.take_profit_manager.calculate_take_profit_levels(
            entry_price, stop_loss, position_type, volatility_factor
        )
        
        # Vérifier le risque de corrélation si une matrice est fournie
        correlation_risk = {}
        if correlation_matrix is not None:
            # Créer une position temporaire pour l'analyse
            temp_position = {
                "symbol": symbol,
                "position_value": position_info["position_value"],
                "risk_amount": position_info["risk_amount"]
            }
            
            # Ajouter la position temporaire aux positions actuelles
            temp_positions = current_positions + [temp_position]
            
            # Vérifier le risque de corrélation
            correlation_risk = self.portfolio_risk_manager.check_correlation_risk(
                temp_positions, correlation_matrix
            )
        
        # Calculer le nombre maximum de positions recommandé
        max_positions = self.portfolio_risk_manager.calculate_max_positions(
            self.portfolio_risk_manager.drawdown_history[-1] if self.portfolio_risk_manager.drawdown_history else 0,
            volatility_factor,
            self.win_rate
        )
        
        # Assembler tous les paramètres
        trade_parameters = {
            "symbol": symbol,
            "entry_price": entry_price,
            "position_type": position_type,
            "stop_loss": stop_loss,
            "take_profit_levels": take_profit_info["levels"],
            "take_profit_distribution": take_profit_info["distribution"],
            "position_value": position_info["position_value"],
            "units": position_info["units"],
            "risk_amount": position_info["risk_amount"],
            "risk_percentage": position_info["risk_percentage"],
            "leverage": leverage,
            "risk_level": self.risk_level.value,
            "risk_adjustment": risk_adjustment,
            "max_positions": max_positions,
            "correlation_risk": correlation_risk
        }
        
        return trade_parameters


def create_default_risk_manager(initial_capital: float = 10000.0) -> RiskManager:
    """
    Crée un gestionnaire de risque avec des paramètres par défaut.
    
    Args:
        initial_capital: Capital initial
        
    Returns:
        RiskManager: Gestionnaire de risque par défaut
    """
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        risk_level=RiskLevel.MEDIUM,
        auto_adjust_risk=True
    )
    
    # Configurer le dimensionnement des positions
    risk_manager.position_sizing = PositionSizing(
        base_risk_per_trade=0.01,
        max_risk_per_trade=0.03,
        max_portfolio_risk=0.10,
        max_correlated_risk=0.15,
        min_position_size=0.01,
        max_position_size=0.20
    )
    
    # Configurer la gestion des stop loss
    risk_manager.stop_loss_manager = StopLossManager(
        atr_multiplier=2.0,
        min_stop_distance=0.01,
        max_stop_distance=0.10,
        trailing_activation=0.02,
        trailing_step=0.005
    )
    
    # Configurer la gestion des prises de profit
    risk_manager.take_profit_manager = TakeProfitManager(
        risk_reward_ratios=[1.5, 2.5, 4.0],
        profit_distribution=[0.3, 0.3, 0.4],
        min_first_target=0.01,
        adjust_for_volatility=True
    )
    
    # Configurer la gestion de l'effet de levier
    risk_manager.leverage_manager = LeverageManager(
        base_leverage=3.0,
        min_leverage=1.0,
        max_leverage=10.0,
        volatility_adjustment=True,
        performance_adjustment=True
    )
    
    # Configurer la gestion du risque du portefeuille
    risk_manager.portfolio_risk_manager = PortfolioRiskManager(
        max_portfolio_risk=0.10,
        max_correlated_risk=0.15,
        max_drawdown_threshold=0.15,
        risk_reduction_factor=0.5,
        correlation_threshold=0.7
    )
    
    return risk_manager


if __name__ == "__main__":
    # Exemple d'utilisation
    risk_manager = create_default_risk_manager(initial_capital=10000.0)
    
    # Simuler quelques trades
    for i in range(10):
        # Simuler un résultat de trade (alternance de gains et pertes)
        profit = 100 if i % 2 == 0 else -50
        trade_result = {
            "profit": profit,
            "capital": risk_manager.current_capital
        }
        
        # Mettre à jour les statistiques
        risk_manager.update_performance_stats(trade_result)
        
        # Mettre à jour le capital
        risk_manager.update_capital(risk_manager.current_capital + profit)
    
    # Calculer les paramètres pour un nouveau trade
    trade_params = risk_manager.calculate_trade_parameters(
        symbol="BTCUSDT_UMCBL",
        entry_price=50000.0,
        position_type="long",
        atr=1000.0,
        volatility_factor=1.2,
        signal_strength=0.7,
        market_regime="trending"
    )
    
    # Afficher les paramètres
    print("Trade Parameters:")
    for key, value in trade_params.items():
        if key != "correlation_risk":  # Éviter d'afficher la structure complexe
            print(f"{key}: {value}")

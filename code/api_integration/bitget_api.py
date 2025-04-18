#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'intégration API pour le système de trading automatisé Bitget.

Ce module implémente les fonctionnalités nécessaires pour interagir avec l'API
de Bitget, y compris l'authentification, la récupération des données de marché,
et l'exécution des ordres.
"""

import os
import sys
import time
import json
import hmac
import base64
import hashlib
import logging
import requests
import websocket
import threading
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from urllib.parse import urlencode

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bitget_api')


class BitgetAPIError(Exception):
    """Exception spécifique pour les erreurs de l'API Bitget."""
    pass


class BitgetAuth:
    """
    Classe pour gérer l'authentification avec l'API Bitget.
    """
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        """
        Initialise l'authentification Bitget.
        
        Args:
            api_key: Clé API Bitget
            api_secret: Secret API Bitget
            passphrase: Passphrase API Bitget
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
    
    def generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Génère une signature pour l'authentification.
        
        Args:
            timestamp: Timestamp en millisecondes
            method: Méthode HTTP (GET, POST, etc.)
            request_path: Chemin de la requête
            body: Corps de la requête (pour les méthodes POST)
            
        Returns:
            Signature encodée en base64
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        d = mac.digest()
        return base64.b64encode(d).decode()
    
    def get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """
        Génère les en-têtes pour une requête authentifiée.
        
        Args:
            method: Méthode HTTP (GET, POST, etc.)
            request_path: Chemin de la requête
            body: Corps de la requête (pour les méthodes POST)
            
        Returns:
            Dictionnaire des en-têtes HTTP
        """
        timestamp = str(int(time.time() * 1000))
        signature = self.generate_signature(timestamp, method, request_path, body)
        
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }


class BitgetRESTClient:
    """
    Client pour l'API REST de Bitget.
    """
    
    BASE_URL = "https://api.bitget.com"
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = ""):
        """
        Initialise le client REST Bitget.
        
        Args:
            api_key: Clé API Bitget (optionnelle pour les endpoints publics)
            api_secret: Secret API Bitget (optionnel pour les endpoints publics)
            passphrase: Passphrase API Bitget (optionnel pour les endpoints publics)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session = requests.Session()
        
        if api_key and api_secret and passphrase:
            self.auth = BitgetAuth(api_key, api_secret, passphrase)
        else:
            self.auth = None
    
    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Effectue une requête à l'API Bitget.
        
        Args:
            method: Méthode HTTP (GET, POST, etc.)
            endpoint: Endpoint de l'API
            params: Paramètres de la requête (pour les méthodes GET)
            data: Données de la requête (pour les méthodes POST)
            
        Returns:
            Réponse de l'API sous forme de dictionnaire
            
        Raises:
            BitgetAPIError: Si l'API retourne une erreur
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Préparer les paramètres de la requête
        if params:
            query_string = urlencode(params)
            request_path = f"{endpoint}?{query_string}"
        else:
            request_path = endpoint
        
        # Préparer le corps de la requête
        body = ""
        if data:
            body = json.dumps(data)
        
        # Préparer les en-têtes
        headers = {}
        if self.auth:
            headers = self.auth.get_headers(method, request_path, body)
        
        # Effectuer la requête
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(url, json=data, headers=headers)
            elif method == "DELETE":
                response = self.session.delete(url, json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            # Vérifier le code de réponse de l'API
            if result.get("code") != "00000":
                raise BitgetAPIError(f"API error: {result.get('msg', 'Unknown error')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise BitgetAPIError(f"Request error: {str(e)}")
    
    # Endpoints Market
    
    def get_symbols(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Récupère toutes les paires de trading disponibles.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Liste des paires disponibles avec leurs informations
        """
        endpoint = "/api/mix/v1/market/contracts"
        params = {"productType": product_type}
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])
    
    def get_tickers(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Récupère les données de ticker pour toutes les paires.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Liste des tickers avec leurs informations
        """
        endpoint = "/api/mix/v1/market/tickers"
        params = {"productType": product_type}
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])
    
    def get_candles(self, symbol: str, granularity: str = "15m", limit: int = 100) -> List[List]:
        """
        Récupère les données de chandeliers pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            granularity: Granularité des chandeliers (1m, 5m, 15m, etc.)
            limit: Nombre de chandeliers à récupérer
            
        Returns:
            Liste des chandeliers [timestamp, open, high, low, close, volume]
        """
        endpoint = "/api/mix/v1/market/candles"
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": limit
        }
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])
    
    def get_open_interest(self, symbol: str, product_type: str = "USDT-FUTURES") -> Dict:
        """
        Récupère l'intérêt ouvert pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Informations sur l'intérêt ouvert
        """
        endpoint = "/api/mix/v1/market/open-interest"
        params = {
            "symbol": symbol,
            "productType": product_type
        }
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", {})
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Récupère le taux de financement actuel pour une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Informations sur le taux de financement
        """
        endpoint = "/api/mix/v1/market/current-funding-rate"
        params = {"symbol": symbol}
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", {})
    
    # Endpoints Account
    
    def get_account(self, symbol: str, margin_coin: str) -> Dict:
        """
        Récupère les informations d'un compte spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Informations sur le compte
        """
        endpoint = "/api/mix/v1/account/account"
        params = {
            "symbol": symbol,
            "marginCoin": margin_coin
        }
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", {})
    
    def get_accounts(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Récupère la liste de tous les comptes.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Liste des comptes avec leurs informations
        """
        endpoint = "/api/mix/v1/account/accounts"
        params = {"productType": product_type}
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])
    
    def set_leverage(self, symbol: str, margin_coin: str, leverage: str, hold_side: str) -> Dict:
        """
        Modifie l'effet de levier pour un symbole spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            leverage: Valeur de l'effet de levier
            hold_side: Direction de la position (long/short)
            
        Returns:
            Résultat de l'opération
        """
        endpoint = "/api/mix/v1/account/set-leverage"
        data = {
            "symbol": symbol,
            "marginCoin": margin_coin,
            "leverage": leverage,
            "holdSide": hold_side
        }
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    def set_margin_mode(self, symbol: str, margin_coin: str, margin_mode: str) -> Dict:
        """
        Modifie le mode de marge (croisé ou isolé).
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            margin_mode: Mode de marge (crossed/fixed)
            
        Returns:
            Résultat de l'opération
        """
        endpoint = "/api/mix/v1/account/set-margin-mode"
        data = {
            "symbol": symbol,
            "marginCoin": margin_coin,
            "marginMode": margin_mode
        }
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    # Endpoints Trade
    
    def place_order(self, symbol: str, margin_coin: str, size: str, side: str, 
                   order_type: str, price: str = None, client_oid: str = None,
                   time_in_force: str = "normal", reduce_only: bool = False) -> Dict:
        """
        Place un ordre sur le marché.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            size: Taille de l'ordre
            side: Direction (buy/sell)
            order_type: Type d'ordre (limit/market)
            price: Prix de l'ordre (requis pour les ordres limites)
            client_oid: ID client optionnel
            time_in_force: Durée de validité (normal/postOnly/ioc/fok)
            reduce_only: Si l'ordre doit uniquement réduire la position
            
        Returns:
            Informations sur l'ordre placé
        """
        endpoint = "/api/mix/v1/order/placeOrder"
        data = {
            "symbol": symbol,
            "marginCoin": margin_coin,
            "size": size,
            "side": side,
            "orderType": order_type,
            "timeInForceValue": time_in_force,
            "reduceOnly": reduce_only
        }
        
        if price:
            data["price"] = price
        
        if client_oid:
            data["clientOid"] = client_oid
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    def cancel_order(self, symbol: str, order_id: str = None, client_oid: str = None) -> Dict:
        """
        Annule un ordre existant.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            order_id: ID de l'ordre à annuler
            client_oid: ID client optionnel
            
        Returns:
            Résultat de l'opération
        """
        endpoint = "/api/mix/v1/order/cancel-order"
        data = {"symbol": symbol}
        
        if order_id:
            data["orderId"] = order_id
        
        if client_oid:
            data["clientOid"] = client_oid
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    def cancel_all_orders(self, symbol: str, margin_coin: str, product_type: str = "USDT-FUTURES") -> Dict:
        """
        Annule tous les ordres pour un symbole spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            product_type: Type de produit (USDT-FUTURES par défaut)
            
        Returns:
            Résultat de l'opération
        """
        endpoint = "/api/mix/v1/order/cancel-all-orders"
        data = {
            "symbol": symbol,
            "marginCoin": margin_coin,
            "productType": product_type
        }
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    def close_all_positions(self, product_type: str = "USDT-FUTURES", margin_coin: str = "USDT") -> Dict:
        """
        Ferme toutes les positions ouvertes.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        endpoint = "/api/mix/v1/position/close-all-position"
        data = {
            "productType": product_type,
            "marginCoin": margin_coin
        }
        
        result = self._request("POST", endpoint, data=data)
        return result.get("data", {})
    
    def get_open_orders(self, symbol: str) -> List[Dict]:
        """
        Récupère les ordres ouverts pour un symbole spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Liste des ordres ouverts
        """
        endpoint = "/api/mix/v1/order/current"
        params = {"symbol": symbol}
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])
    
    def get_positions(self, product_type: str = "USDT-FUTURES", margin_coin: str = "USDT") -> List[Dict]:
        """
        Récupère toutes les positions ouvertes.
        
        Args:
            product_type: Type de produit (USDT-FUTURES par défaut)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Liste des positions ouvertes
        """
        endpoint = "/api/mix/v1/position/all-position"
        params = {
            "productType": product_type,
            "marginCoin": margin_coin
        }
        
        result = self._request("GET", endpoint, params=params)
        return result.get("data", [])


class BitgetWebSocketClient:
    """
    Client pour l'API WebSocket de Bitget.
    """
    
    WS_URL = "wss://ws.bitget.com/mix/v1/stream"
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = ""):
        """
        Initialise le client WebSocket Bitget.
        
        Args:
            api_key: Clé API Bitget (optionnelle pour les canaux publics)
            api_secret: Secret API Bitget (optionnel pour les canaux publics)
            passphrase: Passphrase API Bitget (optionnel pour les canaux publics)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        self.ws = None
        self.thread = None
        self.running = False
        self.ping_interval = 20  # secondes
        self.last_ping = 0
        
        self.callbacks = {
            "ticker": [],
            "candle": [],
            "depth": [],
            "account": [],
            "positions": [],
            "orders": [],
            "error": [],
            "connected": [],
            "disconnected": []
        }
        
        if api_key and api_secret and passphrase:
            self.auth = BitgetAuth(api_key, api_secret, passphrase)
        else:
            self.auth = None
    
    def _generate_auth_message(self) -> Dict:
        """
        Génère un message d'authentification pour WebSocket.
        
        Returns:
            Message d'authentification
        """
        if not self.auth:
            return {}
        
        timestamp = str(int(time.time() * 1000))
        signature = self.auth.generate_signature(timestamp, "GET", "/user/verify", "")
        
        return {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        }
    
    def _on_message(self, ws, message):
        """
        Gère les messages reçus du WebSocket.
        
        Args:
            ws: Instance WebSocketApp
            message: Message reçu
        """
        try:
            data = json.loads(message)
            
            # Gérer les pings
            if "event" in data and data["event"] == "ping":
                self._send_pong()
                return
            
            # Gérer les messages d'erreur
            if "code" in data and data["code"] != "00000":
                logger.error(f"WebSocket error: {data}")
                for callback in self.callbacks["error"]:
                    callback(data)
                return
            
            # Gérer les messages de données
            if "arg" in data and "data" in data:
                channel = data["arg"].get("channel")
                if channel in self.callbacks:
                    for callback in self.callbacks[channel]:
                        callback(data["data"])
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _on_error(self, ws, error):
        """
        Gère les erreurs WebSocket.
        
        Args:
            ws: Instance WebSocketApp
            error: Erreur
        """
        logger.error(f"WebSocket error: {str(error)}")
        for callback in self.callbacks["error"]:
            callback({"error": str(error)})
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Gère la fermeture de la connexion WebSocket.
        
        Args:
            ws: Instance WebSocketApp
            close_status_code: Code de statut de fermeture
            close_msg: Message de fermeture
        """
        logger.info("WebSocket connection closed")
        self.running = False
        
        for callback in self.callbacks["disconnected"]:
            callback({"code": close_status_code, "message": close_msg})
    
    def _on_open(self, ws):
        """
        Gère l'ouverture de la connexion WebSocket.
        
        Args:
            ws: Instance WebSocketApp
        """
        logger.info("WebSocket connection opened")
        self.running = True
        self.last_ping = time.time()
        
        # S'authentifier si nécessaire
        if self.auth:
            auth_message = self._generate_auth_message()
            ws.send(json.dumps(auth_message))
        
        for callback in self.callbacks["connected"]:
            callback({})
    
    def _send_ping(self):
        """
        Envoie un ping au serveur WebSocket.
        """
        if self.ws and self.running:
            self.ws.send(json.dumps({"op": "ping"}))
            self.last_ping = time.time()
    
    def _send_pong(self):
        """
        Envoie un pong en réponse à un ping du serveur.
        """
        if self.ws and self.running:
            self.ws.send(json.dumps({"op": "pong"}))
    
    def _ping_thread(self):
        """
        Thread pour envoyer des pings périodiques.
        """
        while self.running:
            if time.time() - self.last_ping > self.ping_interval:
                self._send_ping()
            time.sleep(1)
    
    def connect(self):
        """
        Établit une connexion WebSocket.
        """
        import websocket as ws_module
        
        if self.running:
            logger.warning("WebSocket already connected")
            return
        
        self.ws = ws_module.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
        
        # Démarrer le thread de ping
        ping_thread = threading.Thread(target=self._ping_thread)
        ping_thread.daemon = True
        ping_thread.start()
        
        # Attendre que la connexion soit établie
        timeout = 5
        start_time = time.time()
        while not self.running and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.running:
            raise BitgetAPIError("Failed to establish WebSocket connection")
    
    def disconnect(self):
        """
        Ferme la connexion WebSocket.
        """
        if self.ws and self.running:
            self.running = False
            self.ws.close()
            
            # Attendre que le thread se termine
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1)
    
    def subscribe(self, instType: str, channel: str, instId: str) -> None:
        """
        S'abonne à un canal WebSocket.
        
        Args:
            instType: Type d'instrument (ex: MC pour Mix Contract)
            channel: Canal (ticker, candle, depth, etc.)
            instId: ID de l'instrument (ex: BTCUSDT)
        """
        if not self.running:
            self.connect()
        
        subscription = {
            "op": "subscribe",
            "args": [{
                "instType": instType,
                "channel": channel,
                "instId": instId
            }]
        }
        
        self.ws.send(json.dumps(subscription))
    
    def subscribe_multiple(self, subscriptions: List[Dict]) -> None:
        """
        S'abonne à plusieurs canaux WebSocket.
        
        Args:
            subscriptions: Liste de dictionnaires de souscription
                [{"instType": "MC", "channel": "ticker", "instId": "BTCUSDT"}, ...]
        """
        if not self.running:
            self.connect()
        
        subscription = {
            "op": "subscribe",
            "args": subscriptions
        }
        
        self.ws.send(json.dumps(subscription))
    
    def unsubscribe(self, instType: str, channel: str, instId: str) -> None:
        """
        Se désabonne d'un canal WebSocket.
        
        Args:
            instType: Type d'instrument (ex: MC pour Mix Contract)
            channel: Canal (ticker, candle, depth, etc.)
            instId: ID de l'instrument (ex: BTCUSDT)
        """
        if not self.running:
            return
        
        unsubscription = {
            "op": "unsubscribe",
            "args": [{
                "instType": instType,
                "channel": channel,
                "instId": instId
            }]
        }
        
        self.ws.send(json.dumps(unsubscription))
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Enregistre un callback pour un événement.
        
        Args:
            event: Nom de l'événement (ticker, candle, depth, account, positions, orders, error, connected, disconnected)
            callback: Fonction de callback
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event: {event}")


class BitgetTrader:
    """
    Classe principale pour le trading sur Bitget.
    """
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, product_type: str = "USDT-FUTURES"):
        """
        Initialise le trader Bitget.
        
        Args:
            api_key: Clé API Bitget
            api_secret: Secret API Bitget
            passphrase: Passphrase API Bitget
            product_type: Type de produit (USDT-FUTURES par défaut)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.product_type = product_type
        
        # Initialiser les clients API
        self.rest_client = BitgetRESTClient(api_key, api_secret, passphrase)
        self.ws_client = BitgetWebSocketClient(api_key, api_secret, passphrase)
        
        # Données de marché en cache
        self.symbols_info = {}
        self.tickers = {}
        self.positions = {}
        self.orders = {}
        self.accounts = {}
        
        # Callbacks
        self.on_ticker_update = None
        self.on_position_update = None
        self.on_order_update = None
        self.on_account_update = None
        self.on_error = None
    
    def initialize(self) -> None:
        """
        Initialise le trader en récupérant les données nécessaires et en configurant les WebSockets.
        """
        # Récupérer les informations sur les symboles
        symbols = self.rest_client.get_symbols(self.product_type)
        self.symbols_info = {s["symbol"]: s for s in symbols}
        
        # Récupérer les tickers
        tickers = self.rest_client.get_tickers(self.product_type)
        self.tickers = {t["symbol"]: t for t in tickers}
        
        # Récupérer les positions
        positions = self.rest_client.get_positions(self.product_type)
        self.positions = {p["symbol"]: p for p in positions}
        
        # Récupérer les comptes
        accounts = self.rest_client.get_accounts(self.product_type)
        self.accounts = {a["marginCoin"]: a for a in accounts}
        
        # Configurer les WebSockets
        self._setup_websockets()
    
    def _setup_websockets(self) -> None:
        """
        Configure les WebSockets pour les mises à jour en temps réel.
        """
        # Callback pour les mises à jour de ticker
        def on_ticker(data):
            for ticker in data:
                symbol = ticker.get("instId")
                if symbol:
                    self.tickers[symbol] = ticker
                    if self.on_ticker_update:
                        self.on_ticker_update(symbol, ticker)
        
        # Callback pour les mises à jour de position
        def on_position(data):
            for position in data:
                symbol = position.get("instId")
                if symbol:
                    self.positions[symbol] = position
                    if self.on_position_update:
                        self.on_position_update(symbol, position)
        
        # Callback pour les mises à jour d'ordre
        def on_order(data):
            for order in data:
                symbol = order.get("instId")
                order_id = order.get("ordId")
                if symbol and order_id:
                    if symbol not in self.orders:
                        self.orders[symbol] = {}
                    self.orders[symbol][order_id] = order
                    if self.on_order_update:
                        self.on_order_update(symbol, order)
        
        # Callback pour les mises à jour de compte
        def on_account(data):
            for account in data:
                margin_coin = account.get("marginCoin")
                if margin_coin:
                    self.accounts[margin_coin] = account
                    if self.on_account_update:
                        self.on_account_update(margin_coin, account)
        
        # Callback pour les erreurs
        def on_error(data):
            logger.error(f"WebSocket error: {data}")
            if self.on_error:
                self.on_error(data)
        
        # Enregistrer les callbacks
        self.ws_client.on("ticker", on_ticker)
        self.ws_client.on("positions", on_position)
        self.ws_client.on("orders", on_order)
        self.ws_client.on("account", on_account)
        self.ws_client.on("error", on_error)
        
        # Connecter le WebSocket
        self.ws_client.connect()
        
        # S'abonner aux canaux
        subscriptions = []
        
        # S'abonner aux tickers pour tous les symboles
        for symbol in self.symbols_info.keys():
            subscriptions.append({
                "instType": "MC",
                "channel": "ticker",
                "instId": symbol
            })
        
        # S'abonner aux positions, ordres et comptes
        subscriptions.extend([
            {"instType": "MC", "channel": "positions", "instId": "default"},
            {"instType": "MC", "channel": "orders", "instId": "default"},
            {"instType": "MC", "channel": "account", "instId": "default"}
        ])
        
        self.ws_client.subscribe_multiple(subscriptions)
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Récupère les informations sur un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Informations sur le symbole
        """
        if symbol in self.symbols_info:
            return self.symbols_info[symbol]
        
        # Si le symbole n'est pas en cache, le récupérer
        symbols = self.rest_client.get_symbols(self.product_type)
        self.symbols_info = {s["symbol"]: s for s in symbols}
        
        return self.symbols_info.get(symbol, {})
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère le ticker pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Ticker du symbole
        """
        if symbol in self.tickers:
            return self.tickers[symbol]
        
        # Si le ticker n'est pas en cache, le récupérer
        tickers = self.rest_client.get_tickers(self.product_type)
        self.tickers = {t["symbol"]: t for t in tickers}
        
        return self.tickers.get(symbol, {})
    
    def get_position(self, symbol: str) -> Dict:
        """
        Récupère la position pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Position du symbole
        """
        if symbol in self.positions:
            return self.positions[symbol]
        
        # Si la position n'est pas en cache, la récupérer
        positions = self.rest_client.get_positions(self.product_type)
        self.positions = {p["symbol"]: p for p in positions}
        
        return self.positions.get(symbol, {})
    
    def get_account(self, margin_coin: str = "USDT") -> Dict:
        """
        Récupère les informations d'un compte.
        
        Args:
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Informations sur le compte
        """
        if margin_coin in self.accounts:
            return self.accounts[margin_coin]
        
        # Si le compte n'est pas en cache, le récupérer
        accounts = self.rest_client.get_accounts(self.product_type)
        self.accounts = {a["marginCoin"]: a for a in accounts}
        
        return self.accounts.get(margin_coin, {})
    
    def place_market_order(self, symbol: str, side: str, size: str, margin_coin: str = "USDT",
                          reduce_only: bool = False, client_oid: str = None) -> Dict:
        """
        Place un ordre au marché.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            side: Direction (buy/sell)
            size: Taille de l'ordre
            margin_coin: Monnaie de marge (ex: USDT)
            reduce_only: Si l'ordre doit uniquement réduire la position
            client_oid: ID client optionnel
            
        Returns:
            Informations sur l'ordre placé
        """
        return self.rest_client.place_order(
            symbol=symbol,
            margin_coin=margin_coin,
            size=size,
            side=side,
            order_type="market",
            reduce_only=reduce_only,
            client_oid=client_oid
        )
    
    def place_limit_order(self, symbol: str, side: str, size: str, price: str, margin_coin: str = "USDT",
                         time_in_force: str = "normal", reduce_only: bool = False, client_oid: str = None) -> Dict:
        """
        Place un ordre limite.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            side: Direction (buy/sell)
            size: Taille de l'ordre
            price: Prix de l'ordre
            margin_coin: Monnaie de marge (ex: USDT)
            time_in_force: Durée de validité (normal/postOnly/ioc/fok)
            reduce_only: Si l'ordre doit uniquement réduire la position
            client_oid: ID client optionnel
            
        Returns:
            Informations sur l'ordre placé
        """
        return self.rest_client.place_order(
            symbol=symbol,
            margin_coin=margin_coin,
            size=size,
            side=side,
            order_type="limit",
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            client_oid=client_oid
        )
    
    def cancel_order(self, symbol: str, order_id: str = None, client_oid: str = None) -> Dict:
        """
        Annule un ordre existant.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            order_id: ID de l'ordre à annuler
            client_oid: ID client optionnel
            
        Returns:
            Résultat de l'opération
        """
        return self.rest_client.cancel_order(
            symbol=symbol,
            order_id=order_id,
            client_oid=client_oid
        )
    
    def cancel_all_orders(self, symbol: str, margin_coin: str = "USDT") -> Dict:
        """
        Annule tous les ordres pour un symbole spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        return self.rest_client.cancel_all_orders(
            symbol=symbol,
            margin_coin=margin_coin,
            product_type=self.product_type
        )
    
    def close_position(self, symbol: str, margin_coin: str = "USDT") -> Dict:
        """
        Ferme la position pour un symbole spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        # Récupérer la position actuelle
        position = self.get_position(symbol)
        
        if not position or float(position.get("total", 0)) == 0:
            return {"success": True, "message": "No position to close"}
        
        # Déterminer la direction et la taille
        position_side = position.get("holdSide")
        position_size = position.get("total")
        
        if position_side == "long":
            side = "sell"
        elif position_side == "short":
            side = "buy"
        else:
            return {"success": False, "message": "Unknown position side"}
        
        # Placer un ordre au marché pour fermer la position
        return self.place_market_order(
            symbol=symbol,
            side=side,
            size=position_size,
            margin_coin=margin_coin,
            reduce_only=True
        )
    
    def close_all_positions(self, margin_coin: str = "USDT") -> Dict:
        """
        Ferme toutes les positions ouvertes.
        
        Args:
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        return self.rest_client.close_all_positions(
            product_type=self.product_type,
            margin_coin=margin_coin
        )
    
    def set_leverage(self, symbol: str, leverage: int, margin_coin: str = "USDT") -> Dict:
        """
        Définit l'effet de levier pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            leverage: Valeur de l'effet de levier
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        # Définir le levier pour les deux directions
        long_result = self.rest_client.set_leverage(
            symbol=symbol,
            margin_coin=margin_coin,
            leverage=str(leverage),
            hold_side="long"
        )
        
        short_result = self.rest_client.set_leverage(
            symbol=symbol,
            margin_coin=margin_coin,
            leverage=str(leverage),
            hold_side="short"
        )
        
        return {
            "long": long_result,
            "short": short_result
        }
    
    def set_margin_mode(self, symbol: str, margin_mode: str, margin_coin: str = "USDT") -> Dict:
        """
        Définit le mode de marge pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            margin_mode: Mode de marge (crossed/fixed)
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Résultat de l'opération
        """
        return self.rest_client.set_margin_mode(
            symbol=symbol,
            margin_coin=margin_coin,
            margin_mode=margin_mode
        )
    
    def get_historical_candles(self, symbol: str, granularity: str = "15m", limit: int = 100) -> pd.DataFrame:
        """
        Récupère les données historiques de chandeliers et les convertit en DataFrame.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            granularity: Granularité des chandeliers (1m, 5m, 15m, etc.)
            limit: Nombre de chandeliers à récupérer
            
        Returns:
            DataFrame avec les données OHLCV
        """
        candles = self.rest_client.get_candles(symbol, granularity, limit)
        
        if not candles:
            return pd.DataFrame()
        
        # Convertir en DataFrame
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convertir les types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Trier par timestamp
        df = df.sort_values("timestamp")
        
        return df
    
    def calculate_order_price(self, symbol: str, price: float) -> float:
        """
        Calcule un prix valide pour un ordre en fonction des règles de l'échange.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            price: Prix souhaité
            
        Returns:
            Prix valide pour l'ordre
        """
        symbol_info = self.get_symbol_info(symbol)
        
        if not symbol_info:
            return price
        
        # Récupérer les informations de précision
        price_place = int(symbol_info.get("pricePlace", 2))
        price_end_step = int(symbol_info.get("priceEndStep", 0))
        
        # Arrondir au nombre de décimales
        rounded_price = round(price, price_place)
        
        # Ajuster en fonction du priceEndStep si nécessaire
        if price_end_step > 0:
            # Convertir en chaîne pour manipuler les décimales
            price_str = f"{rounded_price:.{price_place}f}"
            
            # Extraire la partie décimale
            if "." in price_str:
                integer_part, decimal_part = price_str.split(".")
                
                # Ajuster le dernier chiffre
                if len(decimal_part) > 0:
                    last_digit = int(decimal_part[-1])
                    nearest_step = (last_digit // price_end_step) * price_end_step
                    
                    # Reconstruire le prix
                    if len(decimal_part) > 1:
                        new_decimal = decimal_part[:-1] + str(nearest_step)
                    else:
                        new_decimal = str(nearest_step)
                    
                    adjusted_price = float(f"{integer_part}.{new_decimal}")
                    return adjusted_price
        
        return rounded_price
    
    def calculate_order_size(self, symbol: str, size: float) -> float:
        """
        Calcule une taille valide pour un ordre en fonction des règles de l'échange.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            size: Taille souhaitée
            
        Returns:
            Taille valide pour l'ordre
        """
        symbol_info = self.get_symbol_info(symbol)
        
        if not symbol_info:
            return size
        
        # Récupérer les informations de précision
        size_place = int(symbol_info.get("sizePlace", 0))
        
        # Arrondir au nombre de décimales
        rounded_size = round(size, size_place)
        
        return rounded_size
    
    def get_account_balance(self, margin_coin: str = "USDT") -> float:
        """
        Récupère le solde disponible d'un compte.
        
        Args:
            margin_coin: Monnaie de marge (ex: USDT)
            
        Returns:
            Solde disponible
        """
        account = self.get_account(margin_coin)
        
        if not account:
            return 0.0
        
        # Récupérer le solde disponible
        available = account.get("available", "0")
        
        return float(available)
    
    def get_position_value(self, symbol: str) -> float:
        """
        Récupère la valeur d'une position.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Valeur de la position
        """
        position = self.get_position(symbol)
        
        if not position:
            return 0.0
        
        # Récupérer la taille et le prix de la position
        total = float(position.get("total", "0"))
        avg_price = float(position.get("averageOpenPrice", "0"))
        
        return total * avg_price
    
    def get_position_pnl(self, symbol: str) -> float:
        """
        Récupère le profit/perte d'une position.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Profit/perte de la position
        """
        position = self.get_position(symbol)
        
        if not position:
            return 0.0
        
        # Récupérer le profit/perte non réalisé
        unrealized_pnl = position.get("unrealizedPL", "0")
        
        return float(unrealized_pnl)
    
    def get_position_side(self, symbol: str) -> str:
        """
        Récupère la direction d'une position.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Direction de la position ('long', 'short' ou 'none')
        """
        position = self.get_position(symbol)
        
        if not position or float(position.get("total", 0)) == 0:
            return "none"
        
        return position.get("holdSide", "none")
    
    def get_position_size(self, symbol: str) -> float:
        """
        Récupère la taille d'une position.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Taille de la position
        """
        position = self.get_position(symbol)
        
        if not position:
            return 0.0
        
        # Récupérer la taille de la position
        total = position.get("total", "0")
        
        return float(total)
    
    def get_open_orders(self, symbol: str) -> List[Dict]:
        """
        Récupère les ordres ouverts pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            
        Returns:
            Liste des ordres ouverts
        """
        return self.rest_client.get_open_orders(symbol)
    
    def subscribe_to_ticker(self, symbol: str) -> None:
        """
        S'abonne aux mises à jour de ticker pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
        """
        self.ws_client.subscribe("MC", "ticker", symbol)
    
    def subscribe_to_candles(self, symbol: str, granularity: str = "15m") -> None:
        """
        S'abonne aux mises à jour de chandeliers pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
            granularity: Granularité des chandeliers (1m, 5m, 15m, etc.)
        """
        self.ws_client.subscribe("MC", f"candle{granularity}", symbol)
    
    def subscribe_to_depth(self, symbol: str) -> None:
        """
        S'abonne aux mises à jour de profondeur pour un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: BTCUSDT_UMCBL)
        """
        self.ws_client.subscribe("MC", "depth", symbol)
    
    def cleanup(self) -> None:
        """
        Nettoie les ressources utilisées par le trader.
        """
        if self.ws_client:
            self.ws_client.disconnect()


def create_bitget_trader(api_key: str, api_secret: str, passphrase: str, product_type: str = "USDT-FUTURES") -> BitgetTrader:
    """
    Crée et initialise un trader Bitget.
    
    Args:
        api_key: Clé API Bitget
        api_secret: Secret API Bitget
        passphrase: Passphrase API Bitget
        product_type: Type de produit (USDT-FUTURES par défaut)
        
    Returns:
        BitgetTrader: Trader Bitget initialisé
    """
    trader = BitgetTrader(api_key, api_secret, passphrase, product_type)
    trader.initialize()
    return trader


if __name__ == "__main__":
    # Exemple d'utilisation
    import os
    
    # Récupérer les clés API depuis les variables d'environnement
    api_key = os.environ.get("BITGET_API_KEY", "")
    api_secret = os.environ.get("BITGET_API_SECRET", "")
    passphrase = os.environ.get("BITGET_PASSPHRASE", "")
    
    if not api_key or not api_secret or not passphrase:
        print("Please set BITGET_API_KEY, BITGET_API_SECRET, and BITGET_PASSPHRASE environment variables")
        exit(1)
    
    # Créer un client REST
    rest_client = BitgetRESTClient(api_key, api_secret, passphrase)
    
    # Récupérer les symboles disponibles
    symbols = rest_client.get_symbols()
    print(f"Available symbols: {len(symbols)}")
    
    # Récupérer les tickers
    tickers = rest_client.get_tickers()
    print(f"Tickers: {len(tickers)}")
    
    # Récupérer les positions
    positions = rest_client.get_positions()
    print(f"Positions: {len(positions)}")
    
    # Créer un trader
    trader = create_bitget_trader(api_key, api_secret, passphrase)
    
    # Récupérer le solde du compte
    balance = trader.get_account_balance()
    print(f"Account balance: {balance} USDT")
    
    # Nettoyer les ressources
    trader.cleanup()

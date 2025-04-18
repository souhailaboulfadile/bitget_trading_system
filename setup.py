#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de configuration pour le système de trading automatisé Bitget.

Ce script permet d'installer le système et ses dépendances.
"""

from setuptools import setup, find_packages

setup(
    name="bitget_trading_system",
    version="1.0.0",
    description="Système de trading automatisé pour Bitget (futures USDT)",
    author="Manus AI",
    author_email="contact@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.5.0",
        "requests>=2.26.0",
        "websocket-client>=1.2.0",
        "ccxt>=1.60.0",
        "ta-lib-binary>=0.4.0",
        "pandas-ta>=0.3.0",
        "ipywidgets>=7.6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)

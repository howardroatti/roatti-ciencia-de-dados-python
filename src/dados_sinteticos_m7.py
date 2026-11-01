# -*- coding: utf-8 -*-
"""Dados FICTÍCIOS do Módulo VII (gerados por código).

Todas as séries e amostras abaixo são inventadas por funções determinísticas (numpy,
semente fixa), apenas para ilustrar redes recorrentes e redes generativas. Não derivam
de nenhuma base real, de terceiros ou de repositórios públicos.

Funções:
    serie_senoide(n, seed)             -> série (senoide + ruído) para previsão
    janelas(serie, tam)                -> (X, y) com janelas deslizantes
    tarefa_memoria(n, T, seed)         -> sequências em que a classe depende só do 1º passo
    amostras_gaussiana(n, mu, sigma, seed) -> amostras 1-D para o GAN de brinquedo
"""
import numpy as np


def serie_senoide(n=400, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 25) + 0.5 * np.sin(2 * np.pi * t / 60)
            + rng.normal(0, 0.1, n)).astype("float32")


def janelas(serie, tam=20):
    X = np.stack([serie[i:i + tam] for i in range(len(serie) - tam)])
    y = serie[tam:]
    return X[..., None].astype("float32"), y.astype("float32")


def tarefa_memoria(n=1200, T=20, seed=42):
    """Classe = sinal do 1º valor; os T-1 passos seguintes são ruído distrator."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, T, 1)).astype("float32")
    y = rng.integers(0, 2, n)
    X[:, 0, 0] = np.where(y == 1, 1.0, -1.0)
    return X, y.astype("float32")


def amostras_gaussiana(n=2000, mu=4.0, sigma=1.25, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, (n, 1)).astype("float32")

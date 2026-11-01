# -*- coding: utf-8 -*-
"""Dados FICTÍCIOS do Módulo VI (gerados por código).

Todos os registros abaixo são inventados por funções determinísticas (numpy, semente
fixa). Nada aqui foi copiado, amostrado ou derivado de bases reais, de terceiros ou de
repositórios públicos: os valores são sorteios de distribuições escolhidas pelo autor
apenas para ilustrar as técnicas do livro. Não representam imóveis, pessoas ou
empréstimos reais.

Funções:
    gerar_imoveis(n, seed)      -> DataFrame fictício de imóveis (pré-processamento e
                                   seleção de características)
    gerar_emprestimos(n, seed)  -> DataFrame fictício de crédito (Naive Bayes)
"""
import numpy as np
import pandas as pd

SEED_PADRAO = 42


def gerar_imoveis(n=1500, seed=SEED_PADRAO):
    """Imóveis fictícios: colunas categóricas e numéricas, com preço plausível."""
    rng = np.random.default_rng(seed)
    regioes = ["Norte", "Sul", "Leste", "Oeste", "Centro"]
    tipos = ["casa", "apartamento", "sobrado"]
    metodos = ["venda direta", "leilão", "proposta", "sem informação"]

    regiao = rng.choice(regioes, n, p=[0.22, 0.25, 0.2, 0.2, 0.13])
    tipo = rng.choice(tipos, n, p=[0.5, 0.35, 0.15])
    metodo = rng.choice(metodos, n, p=[0.55, 0.1, 0.25, 0.1])
    distancia = np.round(np.clip(rng.gamma(3.0, 4.0, n), 0.5, 45), 1)
    quartos = np.clip(np.round(rng.normal(2.9, 0.9, n)), 1, 6).astype(int)
    banheiros = np.clip(np.round(quartos * 0.55 + rng.normal(0.4, 0.5, n)), 1, 5).astype(int)
    area = np.round(np.where(tipo == "apartamento", rng.gamma(2, 15, n),
                             rng.gamma(3, 130, n)) + 20).astype(int)
    # CEP fictício: correlacionado com a distância ao centro (variável "confundidora")
    cep = np.round(30000 + distancia * 55 + rng.normal(0, 140, n)).astype(int)
    lat = np.round(-23.5 + rng.normal(0, 0.12, n) - distancia * 0.002, 5)
    lon = np.round(-46.6 + rng.normal(0, 0.12, n) + distancia * 0.002, 5)
    idade = np.round(rng.gamma(2.5, 12, n)).astype(int)
    vagas = np.clip(np.round(quartos * 0.45 + rng.normal(0.2, 0.6, n)), 0, 4).astype(int)
    imoveis_bairro = np.round(rng.gamma(4, 1100, n)).astype(int)

    fator_tipo = pd.Series(tipo).map({"casa": 1.0, "apartamento": 0.8, "sobrado": 1.15}).to_numpy()
    preco = (150_000 + 130_000 * quartos + 60_000 * banheiros + 190 * area
             - 9_500 * distancia - 1_800 * idade + 25_000 * vagas) * fator_tipo
    preco = preco * np.exp(rng.normal(0, 0.12, n))
    preco = np.round(np.clip(preco, 90_000, None), -3)

    df = pd.DataFrame({
        "tipo": tipo, "metodo": metodo, "regiao": regiao,
        "quartos": quartos, "distancia_km": distancia, "cep": cep,
        "banheiros": banheiros, "vagas": vagas, "idade_anos": idade, "area_terreno": area,
        "latitude": lat, "longitude": lon,
        "imoveis_bairro": imoveis_bairro, "preco": preco,
    })
    return df


def gerar_emprestimos(n=5000, seed=SEED_PADRAO):
    """Empréstimos fictícios: outcome ('paid off'/'default') e três atributos categóricos."""
    rng = np.random.default_rng(seed)
    purposes = ["debt_consolidation", "credit_card", "home_improvement",
                "small_business", "other"]
    homes = ["RENT", "OWN", "MORTGAGE"]
    emp = ["> 1 Year", "< 1 Year"]
    purpose_ = rng.choice(purposes, n, p=[0.42, 0.2, 0.12, 0.08, 0.18])
    home_ = rng.choice(homes, n, p=[0.4, 0.15, 0.45])
    emp_len_ = rng.choice(emp, n, p=[0.9, 0.1])
    logit = (-0.1
             + pd.Series(purpose_).map({"debt_consolidation": 0.0, "credit_card": -0.25,
                                        "home_improvement": -0.2, "small_business": 0.7,
                                        "other": 0.25}).to_numpy()
             + pd.Series(home_).map({"RENT": 0.2, "OWN": 0.05, "MORTGAGE": -0.2}).to_numpy()
             + np.where(emp_len_ == "< 1 Year", 0.35, 0.0))
    p_default = 1 / (1 + np.exp(-logit))
    outcome = np.where(rng.random(n) < p_default, "default", "paid off")
    return pd.DataFrame({"outcome": outcome, "purpose_": purpose_,
                         "home_": home_, "emp_len_": emp_len_})


if __name__ == "__main__":
    print(gerar_imoveis().head())
    print(gerar_emprestimos().outcome.value_counts(normalize=True))

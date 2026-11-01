# -*- coding: utf-8 -*-
"""Bases FICTÍCIAS dos Módulos III (Pandas) e V (Visualização).

ATENÇÃO: todos os dados gerados aqui são inventados por código (NumPy, semente
fixa). Nenhum registro corresponde a pessoas, clientes, imóveis ou voos reais, e
nada foi copiado de bases públicas ou privadas. Nomes, valores e relações entre
as variáveis servem apenas para fins didáticos.

As funções são determinísticas: a mesma chamada devolve sempre o mesmo
DataFrame. Uso (a partir da pasta ``codigo``)::

    from src.dados_sinteticos_m3m5 import passageiros_ficticios
    passageiros = passageiros_ficticios()
"""
import numpy as np
import pandas as pd

_PRENOMES_F = ["Alma", "Beatrix", "Cora", "Dalila", "Elora", "Fiona", "Gaia",
               "Helena", "Isolda", "Jade", "Kira", "Lavinia", "Mirela", "Nara"]
_PRENOMES_M = ["Abel", "Bruno", "Caio", "Davi", "Elias", "Fabricio", "Gael",
               "Heitor", "Ivo", "Jonas", "Klaus", "Leo", "Mateus", "Nuno"]
# Sobrenomes inventados (não correspondem a famílias reais)
_SOBRENOMES = ["Valdora", "Quentaris", "Morvelli", "Tabrenha", "Zeluca",
               "Ormindo", "Brasquel", "Nivarro", "Lomendes", "Sabiruna",
               "Cardelho", "Urbinaldo", "Perdeza", "Xarolim", "Finhoque",
               "Gorlinda", "Hevidal", "Ibertaz", "Juvenhal", "Kastrolo"]


def _nomes(rng, sexo):
    """Gera nomes fictícios no formato 'Sobrenome, Prenome'."""
    out = []
    for s in sexo:
        lista = _PRENOMES_F if s == "female" else _PRENOMES_M
        out.append(f"{rng.choice(_SOBRENOMES)}, {rng.choice(lista)}")
    return out


def passageiros_ficticios(n=800, semente=42):
    """Passageiros fictícios de uma viagem de navio (Módulo III).

    Colunas: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch,
    Ticket, Fare, Cabin, Embarked. ``Age``, ``Cabin`` e ``Embarked`` possuem
    valores ausentes (NaN), como em dados reais.
    """
    rng = np.random.default_rng(semente)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.64, 0.36])
    age = np.clip(rng.normal(30, 14, size=n), 1, 80).round(0)
    age[rng.random(n) < 0.20] = np.nan
    sibsp = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.68, 0.23, 0.05, 0.02, 0.02])
    parch = rng.choice([0, 1, 2, 3], size=n, p=[0.76, 0.13, 0.09, 0.02])
    base = np.array([85.0, 22.0, 11.0])[pclass - 1]
    fare = (base * rng.lognormal(0, 0.45, size=n)).round(2)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[rng.random(n) < 0.003] = np.nan
    # Probabilidade fictícia de sobrevivência: maior para mulheres e 1ª classe
    logit = (-1.0 + 2.4 * (sex == "female") + 0.63 * (2 - pclass)
             - 0.015 * np.nan_to_num(age, nan=30) + rng.normal(0, 0.4, n))
    survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cabin = np.array([f"{rng.choice(list('ABCDE'))}{rng.integers(1, 90)}"
                      if (p == 1 and rng.random() < 0.8) or rng.random() < 0.05
                      else np.nan for p in pclass], dtype=object)
    ticket = [f"FIC-{rng.integers(1000, 99999)}" for _ in range(n)]
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": _nomes(rng, sex),
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    })


def clientes_cartao_ficticios(n=2000, semente=7):
    """Clientes fictícios de cartão de crédito (Módulo V).

    Colunas: Customer_ID, Customer_Age, Gender ('M'/'F'), Dependent_count,
    Education_Level, Marital_Status, Income_Category, Card_Category,
    Months_on_book, Credit_Limit, Total_Revolving_Bal, Avg_Utilization_Ratio.
    """
    rng = np.random.default_rng(semente)
    gender = rng.choice(["M", "F"], size=n, p=[0.47, 0.53])
    idade = np.clip(rng.normal(46, 8, n), 21, 73).round().astype(int)
    renda_cat = np.array(["< R$ 3 mil", "R$ 3-6 mil", "R$ 6-9 mil", "R$ 9-12 mil", "> R$ 12 mil"])
    renda_idx = np.clip(rng.normal(1.6 + 0.9 * (gender == "M"), 1.0), 0, 4).round().astype(int)
    renda_base = np.array([2000, 4500, 7500, 10500, 16000])[renda_idx]
    limite = renda_base * rng.lognormal(0.9, 0.45, n) * (1 + 0.15 * (gender == "M"))
    limite = np.clip(limite, 1500, 45000).round(2)
    rev = (limite * np.clip(rng.beta(1.2, 3.0, n), 0, 1)).round(2)
    return pd.DataFrame({
        "Customer_ID": np.arange(100001, 100001 + n),
        "Customer_Age": idade,
        "Gender": gender,
        "Dependent_count": rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.09, 0.18, 0.27, 0.27, 0.15, 0.04]),
        "Education_Level": rng.choice(["Ensino Médio", "Graduação", "Pós-graduação", "Mestrado", "Doutorado"],
                                      size=n, p=[0.25, 0.40, 0.20, 0.10, 0.05]),
        "Marital_Status": rng.choice(["Casado", "Solteiro", "Divorciado"], size=n, p=[0.47, 0.42, 0.11]),
        "Income_Category": renda_cat[renda_idx],
        "Card_Category": rng.choice(["Basic", "Silver", "Gold", "Platinum"], size=n, p=[0.80, 0.14, 0.05, 0.01]),
        "Months_on_book": np.clip(idade - 18 - rng.integers(0, 20, n), 12, 56),
        "Credit_Limit": limite,
        "Total_Revolving_Bal": rev,
        "Avg_Utilization_Ratio": (rev / limite).round(3),
    })


def vendas_casas_ficticias(n=1500, semente=11):
    """Vendas de imóveis fictícias (Módulo V).

    Colunas: SalePrice, zhvi_px, zhvi_idx, AdjSalePrice, NbrLivingUnits, SqFtLot,
    SqFtTotLiving, SqFtFinBasement, Bathrooms, Bedrooms, BldgGrade, YrBuilt,
    YrRenovated, TrafficNoise, LandVal, ImpsVal. Valores em dólares e pés
    quadrados, inventados.
    """
    rng = np.random.default_rng(semente)
    area = np.clip(rng.lognormal(7.5, 0.35, n), 600, 7000).round()
    quartos = np.clip((area / 700 + rng.normal(0, 0.8, n)).round(), 1, 8).astype(int)
    banh = np.clip((quartos * 0.6 + rng.normal(0.5, 0.5, n)) * 2, 1, 8).round() / 2
    nota = np.clip((6 + (area - 2000) / 900 + rng.normal(0, 0.8, n)).round(), 3, 13).astype(int)
    lote = np.clip(area * rng.lognormal(1.2, 0.6, n), 1000, 80000).round()
    sub = np.where(rng.random(n) < 0.35, (area * rng.uniform(0.1, 0.5, n)).round(), 0)
    construido = rng.integers(1900, 2015, n)
    reno = np.where((rng.random(n) < 0.12) & (construido < 2000), rng.integers(1990, 2015, n), 0)
    ruido = rng.choice([0, 1, 2, 3], size=n, p=[0.62, 0.22, 0.12, 0.04])
    idx_mes = rng.normal(1.0, 0.08, n).round(4)
    zhvi_px = (rng.choice([380000, 420000, 460000], size=n) * idx_mes).round()
    terreno = (lote * 6 + nota * 12000 + rng.normal(0, 25000, n)).clip(20000)
    benfeit = (area * (90 + nota * 18) - 400 * (2015 - construido) + rng.normal(0, 30000, n)).clip(15000)
    preco = ((terreno + benfeit) * (1 - 0.04 * ruido)).round(-2)
    return pd.DataFrame({
        "SalePrice": preco,
        "zhvi_px": zhvi_px,
        "zhvi_idx": idx_mes,
        "AdjSalePrice": (preco / idx_mes).round(-2),
        "NbrLivingUnits": rng.choice([1, 2, 3], size=n, p=[0.93, 0.06, 0.01]),
        "SqFtLot": lote,
        "SqFtTotLiving": area,
        "SqFtFinBasement": sub,
        "Bathrooms": banh,
        "Bedrooms": quartos,
        "BldgGrade": nota,
        "YrBuilt": construido,
        "YrRenovated": reno,
        "TrafficNoise": ruido,
        "LandVal": terreno.round(-2),
        "ImpsVal": benfeit.round(-2),
    })


def voos_ficticios(semente=3):
    """Passageiros mensais de uma companhia aérea fictícia, 1949-1960 (Módulo V).

    Colunas: year, month, passengers (tendência de alta com sazonalidade).
    """
    rng = np.random.default_rng(semente)
    meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    linhas = []
    for ano in range(1949, 1961):
        t = ano - 1949
        for m, nome in enumerate(meses):
            sazonal = 1 + 0.22 * np.sin((m - 3) / 12 * 2 * np.pi)
            linhas.append((ano, nome, int((110 + 28 * t) * sazonal * rng.normal(1, 0.04))))
    df = pd.DataFrame(linhas, columns=["year", "month", "passengers"])
    df["month"] = pd.Categorical(df["month"], categories=meses, ordered=True)
    return df


def gorjetas_ficticias(n=250, semente=5):
    """Contas e gorjetas fictícias de um restaurante (Módulo V).

    Colunas: total_bill, tip, sex, smoker, day, time, size.
    """
    rng = np.random.default_rng(semente)
    size = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.03, 0.55, 0.16, 0.15, 0.08, 0.03])
    conta = np.clip(rng.normal(8 + 7 * size, 4, n), 4, 60).round(2)
    pct = np.clip(rng.normal(0.16, 0.05, n), 0.02, 0.45)
    dia = rng.choice(["Thur", "Fri", "Sat", "Sun"], size=n, p=[0.25, 0.08, 0.37, 0.30])
    return pd.DataFrame({
        "total_bill": conta,
        "tip": (conta * pct).round(2),
        "sex": rng.choice(["Male", "Female"], size=n, p=[0.64, 0.36]),
        "smoker": rng.choice(["Yes", "No"], size=n, p=[0.38, 0.62]),
        "day": pd.Categorical(dia, categories=["Thur", "Fri", "Sat", "Sun"], ordered=True),
        "time": np.where(np.isin(dia, ["Thur", "Fri"]), "Lunch", "Dinner"),
        "size": size,
    })


# ---------------------------------------------------------------------------
# Bases FICTÍCIAS de uma pequena loja (Módulo III: merge, groupby, séries temporais)
# ---------------------------------------------------------------------------
_CIDADES_LOJA = ["Porto Alvo", "Serra Verde", "Vale Claro", "Monte Azul"]
_PRODUTOS_LOJA = [
    ("Caderno Brochura", "Papelaria", 18.90),
    ("Caneta Esferográfica", "Papelaria", 3.50),
    ("Agenda Anual", "Papelaria", 32.00),
    ("Cadeira Giratória", "Escritório", 389.00),
    ("Mesa de Canto", "Escritório", 270.00),
    ("Luminária de Mesa", "Escritório", 79.90),
    ("Teclado Compacto", "Informática", 129.00),
    ("Mouse Óptico", "Informática", 59.90),
]


def produtos_ficticios():
    """Catálogo fictício de 8 produtos: id_produto, produto, categoria, preco."""
    return pd.DataFrame({
        "id_produto": np.arange(1, len(_PRODUTOS_LOJA) + 1),
        "produto": [p[0] for p in _PRODUTOS_LOJA],
        "categoria": [p[1] for p in _PRODUTOS_LOJA],
        "preco": [p[2] for p in _PRODUTOS_LOJA],
    })


def clientes_loja_ficticios(n=40, semente=21):
    """Cadastro fictício de clientes: id_cliente, nome, cidade, segmento.

    Os nomes têm espaços e caixa irregulares de propósito (para praticar
    métodos de string).
    """
    rng = np.random.default_rng(semente)
    nomes = []
    for _ in range(n):
        prenome = rng.choice(_PRENOMES_F + _PRENOMES_M)
        sobrenome = rng.choice(_SOBRENOMES)
        nome = f"{prenome} {sobrenome}"
        sorteio = rng.integers(0, 3)
        nomes.append(nome.upper() if sorteio == 0 else
                     (f"  {nome.lower()} " if sorteio == 1 else nome))
    return pd.DataFrame({
        "id_cliente": np.arange(1, n + 1),
        "nome": nomes,
        "cidade": rng.choice(_CIDADES_LOJA, size=n),
        "segmento": rng.choice(["Varejo", "Empresa"], size=n, p=[0.7, 0.3]),
    })


def vendas_ficticias(n=600, semente=21):
    """Vendas fictícias de 2024: id_venda, data, id_cliente, id_produto,
    quantidade, desconto (com NaN), canal.

    Alguns ``id_cliente`` (41 a 45) não existem em ``clientes_loja_ficticios()``,
    e há clientes sem nenhuma venda: útil para comparar os tipos de ``merge``.
    """
    rng = np.random.default_rng(semente)
    dias = rng.integers(0, 366, size=n)
    # mais vendas no fim do ano (sazonalidade fictícia)
    dias = np.where(rng.random(n) < 0.25, rng.integers(300, 366, size=n), dias)
    data = pd.Timestamp("2024-01-01") + pd.to_timedelta(dias, unit="D")
    desconto = rng.choice([0.0, 0.05, 0.10, 0.15], size=n, p=[0.55, 0.2, 0.15, 0.1])
    desconto[rng.random(n) < 0.12] = np.nan
    df = pd.DataFrame({
        "id_venda": np.arange(1, n + 1),
        "data": data,
        "id_cliente": rng.integers(1, 46, size=n),
        "id_produto": rng.choice(np.arange(1, 9), size=n,
                                 p=[0.2, 0.25, 0.1, 0.05, 0.05, 0.1, 0.12, 0.13]),
        "quantidade": rng.integers(1, 6, size=n),
        "desconto": desconto,
        "canal": rng.choice(["loja", "site", "app"], size=n, p=[0.5, 0.3, 0.2]),
    })
    return df.sort_values("data").reset_index(drop=True).assign(
        id_venda=lambda d: np.arange(1, len(d) + 1))

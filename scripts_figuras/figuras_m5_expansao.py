# -*- coding: utf-8 -*-
"""Figuras das novas seções do Módulo V (escolha do gráfico, honestidade visual,
cores, distribuições, relações, séries temporais, pequenos múltiplos).

Todas usam as bases FICTÍCIAS de ``src/dados_sinteticos_m3m5.py`` (geradas por
código, com semente fixa). O código de cada figura é o mesmo mostrado no texto
do módulo; aqui apenas acrescentamos o ``savefig``.

Rode a partir de qualquer pasta:  python figuras_m5_expansao.py
Saída em ../../figuras (fig_m5_*.png).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m3m5 import (clientes_cartao_ficticios,
                                       gorjetas_ficticias, voos_ficticios)

OUT = os.path.join(AQUI, "..", "..", "figuras")
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --- dados (mesmo bloco do texto) -------------------------------------------
sns.set_theme(style="whitegrid")
gorjetas = gorjetas_ficticias()
gorjetas["pct_gorjeta"] = gorjetas["tip"] / gorjetas["total_bill"] * 100
clientes = clientes_cartao_ficticios()
voos = voos_ficticios()


def eixo_truncado():
    medias = gorjetas.groupby("day", observed=True)["pct_gorjeta"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(medias.index.astype(str), medias.values)
    ax1.set_ylim(medias.min() - 0.5, medias.max() + 0.5)
    ax1.set_title("Eixo truncado: a diferença parece enorme")
    ax2.bar(medias.index.astype(str), medias.values)
    ax2.set_ylim(0, 20)
    ax2.set_title("Eixo a partir do zero: diferença pequena")
    for ax in (ax1, ax2):
        ax.set_ylabel("Gorjeta média (% da conta)")
    fig.tight_layout()
    salvar(fig, "fig_m5_eixo_truncado.png")


def paletas():
    DEUTAN = np.array([[0.367, 0.861, -0.228],
                       [0.280, 0.673, 0.047],
                       [-0.012, 0.043, 0.969]])

    def simular_deuteranopia(cores):
        return np.clip(np.array(cores) @ DEUTAN.T, 0, 1)

    paletas = {"Sequencial (viridis)": sns.color_palette("viridis", 7),
               "Divergente (RdBu)": sns.color_palette("RdBu", 7),
               "Qualitativa (colorblind)": sns.color_palette("colorblind", 7),
               "Vermelho e verde (evitar)": [(0.84, 0.15, 0.16), (0.17, 0.63, 0.17)] * 3 + [(0.84, 0.15, 0.16)]}
    fig, axes = plt.subplots(len(paletas), 2, figsize=(8, 5))
    for linha, (nome, cores) in zip(axes, paletas.items()):
        for ax, c, rotulo in zip(linha, (cores, simular_deuteranopia(cores)),
                                 (nome, "visão simulada (deuteranopia)")):
            ax.imshow([list(range(len(c)))], cmap=ListedColormap(c), aspect="auto")
            ax.set_title(rotulo, loc="left", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    salvar(fig, "fig_m5_paletas.png")


def distribuicoes():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    sns.histplot(data=clientes, x="Credit_Limit", bins=30, ax=axes[0, 0])
    sns.kdeplot(data=clientes, x="Credit_Limit", hue="Gender", fill=True,
                common_norm=False, cut=0, ax=axes[0, 1])
    sns.boxplot(data=clientes, x="Gender", y="Credit_Limit", hue="Gender",
                legend=False, ax=axes[1, 0])
    sns.violinplot(data=clientes, x="Gender", y="Credit_Limit", hue="Gender",
                   inner="quartile", cut=0, legend=False, ax=axes[1, 1])
    axes[0, 0].set_title("Histograma")
    axes[0, 1].set_title("Densidade (KDE) por gênero")
    axes[1, 0].set_title("Boxplot")
    axes[1, 1].set_title("Violin")
    fig.tight_layout()
    salvar(fig, "fig_m5_distribuicoes.png")


def relacao():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=gorjetas, x="total_bill", y="tip", hue="day", alpha=0.7, ax=ax1)
    sns.regplot(data=gorjetas, x="total_bill", y="tip", ci=95, seed=0,
                scatter_kws={"alpha": 0.4}, ax=ax2)
    ax1.set_title("Dispersão colorida por dia")
    ax2.set_title("Reta ajustada e intervalo de 95%")
    fig.tight_layout()
    salvar(fig, "fig_m5_relacao_gorjeta.png")


def serie_temporal():
    voos["data"] = pd.to_datetime(dict(year=voos["year"],
                                       month=voos["month"].cat.codes + 1, day=1))
    serie = voos.set_index("data")["passengers"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(serie.index, serie.values, color="gray", linewidth=1, label="Mensal")
    ax.plot(serie.index, serie.rolling(12).mean(), color="#d9822b", linewidth=2.5,
            label="Média móvel de 12 meses")
    ax.set(title="Passageiros mensais (companhia fictícia)", xlabel="Ano",
           ylabel="Passageiros")
    ax.legend()
    fig.tight_layout()
    salvar(fig, "fig_m5_serie_temporal.png")


def pequenos_multiplos():
    g = sns.FacetGrid(gorjetas, col="day", col_wrap=2, height=3, aspect=1.3)
    g.map_dataframe(sns.histplot, x="pct_gorjeta", bins=15)
    g.set_axis_labels("Gorjeta (% da conta)", "Contagem")
    g.set_titles("{col_name}")
    salvar(g.figure, "fig_m5_pequenos_multiplos.png")


if __name__ == "__main__":
    eixo_truncado()
    paletas()
    distribuicoes()
    relacao()
    serie_temporal()
    pequenos_multiplos()
    print(gorjetas.groupby("day", observed=True)["pct_gorjeta"].mean().round(2).to_dict())
    print(clientes.groupby("Gender")["Credit_Limit"].describe().round(0))
    print(clientes["Credit_Limit"].skew().round(2))
    print(gorjetas[["total_bill", "tip"]].corr().round(3))
    from numpy.polynomial import polynomial as P
    print(np.polyfit(gorjetas.total_bill, gorjetas.tip, 1).round(3))
    print(voos.groupby("year")["passengers"].sum().iloc[[0, -1]].to_dict())

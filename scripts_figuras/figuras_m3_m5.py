# -*- coding: utf-8 -*-
"""Gera as figuras do Módulo V a partir das bases FICTÍCIAS (dados_sinteticos_m3m5).

Rode a partir de qualquer pasta:  python figuras_m3_m5.py
Saída em ../../figuras (fig_m5_boxplot_limite.png, fig_m5_heatmap_correlacao.png).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m3m5 import clientes_cartao_ficticios, vendas_casas_ficticias

OUT = os.path.join(AQUI, "..", "..", "figuras")
AZUL, LARANJA = "#2c6fbf", "#d9822b"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boxplot_limite():
    df = clientes_cartao_ficticios()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bp = ax.boxplot([df[df.Gender == "M"].Credit_Limit, df[df.Gender == "F"].Credit_Limit],
                    tick_labels=["Masculino", "Feminino"], patch_artist=True,
                    flierprops=dict(markerfacecolor="g", marker="D", markersize=4))
    for caixa, cor in zip(bp["boxes"], [AZUL, LARANJA]):
        caixa.set(facecolor=cor, alpha=0.6)
    ax.set_title("Limite de Créditos x Gênero (dados fictícios)")
    ax.set_ylabel("Limite de crédito (R$)")
    ax.grid(axis="y", alpha=0.3)
    salvar(fig, "fig_m5_boxplot_limite.png")


def heatmap_correlacao():
    v = vendas_casas_ficticias()
    corr = v.corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(v.columns)))
    ax.set_yticks(np.arange(len(v.columns)))
    ax.set_xticklabels(v.columns)
    ax.set_yticklabels(v.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlação de Pearson")
    ax.set_title("Correlação entre atributos (vendas de imóveis fictícias)")
    salvar(fig, "fig_m5_heatmap_correlacao.png")


if __name__ == "__main__":
    boxplot_limite()
    heatmap_correlacao()
    df = clientes_cartao_ficticios()
    print(df.groupby("Gender").Credit_Limit.median())
    v = vendas_casas_ficticias()
    print(v.corr()["SalePrice"].round(2))

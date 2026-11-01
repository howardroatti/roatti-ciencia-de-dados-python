# -*- coding: utf-8 -*-
"""Figuras novas dos Módulos II (broadcasting) e III (série temporal com média móvel).

Rode a partir de qualquer pasta:  python figuras_m2_m3_expansao.py
Saída em ../../figuras (fig_m2_broadcasting.png, fig_m3_serie_media_movel.png).
Os dados do Módulo III são FICTÍCIOS (src.dados_sinteticos_m3m5.vendas_ficticias).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m3m5 import vendas_ficticias

OUT = os.path.join(AQUI, "..", "..", "figuras")
AZUL, LARANJA, CINZA = "#2c6fbf", "#d9822b", "#9aa3ad"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grade(ax, valores, x0, y0, cor, virtual=None, tam=0.9):
    """Desenha uma grade de células com os valores (matriz 2D) a partir de (x0, y0).

    ``virtual`` é uma máscara booleana: células "esticadas" (cópias virtuais)
    são desenhadas tracejadas e mais claras.
    """
    nl, nc = valores.shape
    for i in range(nl):
        for j in range(nc):
            v = virtual is not None and virtual[i, j]
            ax.add_patch(Rectangle((x0 + j, y0 - i - 1), tam, tam,
                                   facecolor=cor, alpha=0.25 if v else 0.75,
                                   edgecolor=cor, linestyle="--" if v else "-", linewidth=1.2))
            ax.text(x0 + j + tam / 2, y0 - i - 1 + tam / 2, f"{valores[i, j]:g}",
                    ha="center", va="center", fontsize=9, color="#222222" if v else "white")


def painel(ax, A, B, titulo, formaB):
    """A (matriz) + B (esticado para a forma de A ou do resultado)."""
    R = A + B
    nl, nc = R.shape
    ax.set_title(titulo, fontsize=10)
    # operando A
    grade(ax, np.broadcast_to(A, R.shape), 0, nl, AZUL,
          virtual=np.array([[(A.shape[0] == 1 and i > 0) or (A.shape[1] == 1 and j > 0)
                             for j in range(nl and nc)] for i in range(nl)]))
    ax.text(nc / 2 - 0.05, nl + 0.35, f"A  {A.shape}", ha="center", fontsize=9)
    ox = nc + 1.0
    ax.text(ox - 0.55, nl / 2, "+", ha="center", va="center", fontsize=16)
    virt_b = np.array([[(B.shape[0] == 1 and i > 0) or (B.shape[1] == 1 and j > 0)
                        for j in range(nc)] for i in range(nl)])
    grade(ax, np.broadcast_to(B, R.shape), ox, nl, LARANJA, virtual=virt_b)
    ax.text(ox + nc / 2 - 0.05, nl + 0.35, f"B  {formaB}", ha="center", fontsize=9)
    rx = 2 * nc + 2.4
    ax.text(rx - 0.8, nl / 2, "=", ha="center", va="center", fontsize=16)
    grade(ax, R, rx, nl, CINZA)
    ax.text(rx + nc / 2 - 0.05, nl + 0.35, f"A + B  {R.shape}", ha="center", fontsize=9)
    ax.set_xlim(-0.3, rx + nc + 0.3)
    ax.set_ylim(-0.3, nl + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def broadcasting():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.5]})
    A1 = np.arange(6).reshape(3, 2)
    painel(axes[0], A1, np.array([[10, 20]]), "(3, 2) + (2,)", "(2,)")
    painel(axes[1], A1, np.array([[100], [200], [300]]), "(3, 2) + (3, 1)", "(3, 1)")
    painel(axes[2], np.arange(3).reshape(3, 1), np.arange(4).reshape(1, 4),
           "(3, 1) + (1, 4)", "(1, 4)")
    fig.text(0.5, 0.17, "Células tracejadas e claras são cópias virtuais: o NumPy não "
             "replica os dados na memória.", ha="center", fontsize=9, color="#444444")
    salvar(fig, "fig_m2_broadcasting.png")


def serie_media_movel():
    v = vendas_ficticias()
    diario = v.set_index("data")["quantidade"].resample("D").sum()
    mm = diario.rolling(window=7, center=False).mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(diario.index, diario.values, color=CINZA, linewidth=0.9, label="Unidades por dia")
    ax.plot(mm.index, mm.values, color=AZUL, linewidth=2.2, label="Média móvel de 7 dias")
    ax.set_title("Vendas diárias e média móvel (dados fictícios)")
    ax.set_ylabel("Unidades vendidas")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    salvar(fig, "fig_m3_serie_media_movel.png")


if __name__ == "__main__":
    broadcasting()
    serie_media_movel()
    print("figuras geradas em", os.path.abspath(OUT))

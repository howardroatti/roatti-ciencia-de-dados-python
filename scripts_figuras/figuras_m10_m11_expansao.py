# -*- coding: utf-8 -*-
"""Figuras de expansão dos Módulos X e XI (dados e traços FICTÍCIOS/simulados).

Rode:  python figuras_m10_m11_expansao.py
Saída em ../../figuras: fig_m10_laco_react.png, fig_m11_harness.png
"""
import os, re, sys, unicodedata
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m10_m11 import GOLDEN_SET, RESPOSTAS_SIMULADAS

OUT = os.path.join(AQUI, "..", "..", "figuras")
AZUL, LARANJA, VERDE, CINZA = "#2c6fbf", "#d9822b", "#3a9a5b", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def laco_react():
    """Traço do agente simulado: busca -> cálculo -> resposta final."""
    passos = [
        ("Pensamento", "Preciso do dado\nda população.", AZUL),
        ("Ação", "buscar_base(\n'populacao de\nbrasilia')", LARANJA),
        ("Observação", "2800000", VERDE),
        ("Pensamento", "Falta multiplicar\npor 2.", AZUL),
        ("Ação", "calculadora(\n'2800000*2')", LARANJA),
        ("Observação", "5600000", VERDE),
        ("Resposta final", "5.600.000", CINZA),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 7); ax.set_ylim(0.05, 1.75); ax.axis("off")
    for i, (rotulo, texto, cor) in enumerate(passos):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.55), 0.88, 1.1, boxstyle="round,pad=0.02",
                                    fc=cor, ec="none", alpha=0.9))
        ax.text(i + 0.5, 1.45, rotulo, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        ax.text(i + 0.5, 0.98, texto, ha="center", va="center", color="white", fontsize=7)
        if i < len(passos) - 1:
            ax.annotate("", xy=(i + 1.06, 1.1), xytext=(i + 0.94, 1.1),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(3.5, 0.18, "Laço ReAct: pensar → agir → observar, repetido até a resposta final "
            "(ou até o limite de passos)", ha="center", fontsize=8.5, color="#333")
    salvar(fig, "fig_m10_laco_react.png")


def _norm(t):
    s = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def harness():
    metricas = ["Exata", "Normalizada", "Cobertura de\npalavras-chave"]
    resumo = {}
    cob = {}
    for v in ("v1", "v2"):
        ex, no, co = [], [], []
        for it in GOLDEN_SET:
            r = RESPOSTAS_SIMULADAS[v][it["id"]]
            ex.append(r == it["esperada"])
            no.append(_norm(r) == _norm(it["esperada"]))
            co.append(np.mean([_norm(p) in _norm(r) for p in it["palavras"]]))
        resumo[v] = [np.mean(ex), np.mean(no), np.mean(co)]
        cob[v] = co
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(3); w = 0.36
    a1.bar(x - w / 2, resumo["v1"], w, color=LARANJA, label="prompt v1")
    a1.bar(x + w / 2, resumo["v2"], w, color=AZUL, label="prompt v2")
    for xs, vals in ((x - w / 2, resumo["v1"]), (x + w / 2, resumo["v2"])):
        for xi, vi in zip(xs, vals):
            a1.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(metricas, fontsize=8)
    a1.set_ylim(0, 1.15); a1.set_title("Métricas no golden set (6 perguntas fictícias)", fontsize=10)
    a1.legend(frameon=False, fontsize=8); a1.grid(axis="y", alpha=0.3)
    ids = [it["id"] for it in GOLDEN_SET]
    a2.plot(ids, cob["v1"], "o-", color=LARANJA, label="v1")
    a2.plot(ids, cob["v2"], "s-", color=AZUL, label="v2")
    a2.axhline(0.8, color=CINZA, ls="--", lw=1); a2.text(1, 0.82, "limiar 0,8", fontsize=8, color=CINZA)
    a2.set_xlabel("pergunta (id)"); a2.set_ylabel("cobertura"); a2.set_ylim(-0.05, 1.1)
    a2.set_title("Cobertura por pergunta", fontsize=10); a2.legend(frameon=False, fontsize=8)
    a2.grid(alpha=0.3)
    salvar(fig, "fig_m11_harness.png")


if __name__ == "__main__":
    laco_react()
    harness()
    print("figuras salvas em", os.path.abspath(OUT))

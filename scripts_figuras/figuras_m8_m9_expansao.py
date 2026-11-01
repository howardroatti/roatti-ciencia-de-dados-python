# -*- coding: utf-8 -*-
"""Figuras das seções de expansão dos Módulos VIII e IX (dados FICTÍCIOS, sementes fixas).

Rode a partir de qualquer pasta:  python figuras_m8_m9_expansao.py
Saída em ../../figuras:
    fig_m8_heatmap_atencao.png, fig_m8_posicional.png,
    fig_m9_recuperadores.png, fig_m9_chunking.png, fig_m9_scores.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m8_m9 import (documentos_ficticios, perguntas_ficticias,
                                        perguntas_fora_da_base)

OUT = os.path.join(AQUI, "..", "..", "figuras")
AZUL, LARANJA, VERDE, CINZA = "#2c6fbf", "#d9822b", "#3a9a5b", "#8a8a8a"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ Módulo VIII
def softmax(x, eixo=-1):
    x = x - x.max(axis=eixo, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=eixo, keepdims=True)


def atencao(Q, K, V, mascara=None):
    pont = Q @ K.T / np.sqrt(Q.shape[-1])
    if mascara is not None:
        pont = np.where(mascara, pont, -1e9)
    pesos = softmax(pont)
    return pesos @ V, pesos


def codificacao_posicional(n_pos, d_modelo):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_modelo // 2)[None, :]
    ang = pos / (10000 ** (2 * i / d_modelo))
    PE = np.zeros((n_pos, d_modelo))
    PE[:, 0::2], PE[:, 1::2] = np.sin(ang), np.cos(ang)
    return PE


def heatmap_atencao():
    tokens = ["o", "gato", "comeu", "o", "peixe"]
    n, d_modelo, d_k = len(tokens), 8, 4
    rng = np.random.default_rng(42)                       # mesma sequência do notebook
    tabela = rng.normal(size=(4, d_modelo))
    X = tabela[[0, 1, 2, 0, 3]]
    Wq, Wk, Wv = (rng.normal(size=(d_modelo, d_k)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    _, p_livre = atencao(Q, K, V)
    _, p_causal = atencao(Q, K, V, mascara=np.tril(np.ones((n, n), dtype=bool)))
    fig, eixos = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    for ax, p, titulo in zip(eixos, [p_livre, p_causal],
                             ["Atenção bidirecional (sem máscara)", "Atenção causal (com máscara)"]):
        im = ax.imshow(p, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), tokens)
        ax.set_yticks(range(n), tokens)
        ax.set_xlabel("token consultado (key)")
        ax.set_ylabel("token que consulta (query)")
        ax.set_title(titulo, fontsize=10)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if p[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=eixos, label="peso de atenção", shrink=0.85)
    salvar(fig, "fig_m8_heatmap_atencao.png")


def posicional():
    PE = codificacao_posicional(50, 16)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a1.imshow(PE.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    a1.set_xlabel("posição na sequência")
    a1.set_ylabel("dimensão do vetor")
    a1.set_title("Codificação posicional senoidal", fontsize=10)
    fig.colorbar(im, ax=a1, label="valor")
    pos = np.arange(50)
    for d, cor in zip([0, 4, 8], [AZUL, LARANJA, VERDE]):
        a2.plot(pos, PE[:, d], color=cor, label=f"dimensão {d}")
    a2.set_xlabel("posição na sequência")
    a2.set_title("Frequências decrescentes", fontsize=10)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    salvar(fig, "fig_m8_posicional.png")


# ------------------------------------------------------------------ Módulo IX
STOP = ("a o as os um uma uns umas de da do das dos em no na nos nas por para com sem que "
        "e ou se ao aos mais mas ja nao ser sao foi vale pode podem posso como quando "
        "qual quais quanto quantos quantas onde voces vocês ha meu meus minha minhas").split()


def fragmentar(texto, tamanho=40, sobreposicao=10):
    palavras = texto.split()
    passo = tamanho - sobreposicao
    return [" ".join(palavras[i:i + tamanho])
            for i in range(0, max(len(palavras) - sobreposicao, 1), passo)]


def indexar(docs, tamanho=40, sobreposicao=10):
    return [{"doc": d, "texto": t} for d, texto in docs.items()
            for t in fragmentar(texto, tamanho, sobreposicao)]


class RecuperadorTFIDF:
    def __init__(self, fragmentos, **opcoes):
        self.fragmentos = fragmentos
        if opcoes.get("analyzer") == "char_wb":
            self.vetorizador = TfidfVectorizer(strip_accents="unicode", **opcoes)
        else:
            self.vetorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                               stop_words=STOP, **opcoes)
        self.matriz = self.vetorizador.fit_transform([f["texto"] for f in fragmentos])

    def buscar(self, pergunta, k=3):
        sims = cosine_similarity(self.vetorizador.transform([pergunta]), self.matriz).ravel()
        melhores = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.fragmentos[i]) for i in melhores]


def avaliar(rec, perguntas):
    posicoes = []
    for pergunta, alvo in perguntas:
        ranking = []
        for _, f in rec.buscar(pergunta, k=len(rec.fragmentos)):
            if f["doc"] not in ranking:
                ranking.append(f["doc"])
        posicoes.append(ranking.index(alvo) + 1)
    pos = np.array(posicoes)
    return {"hit@1": float((pos <= 1).mean()), "hit@3": float((pos <= 3).mean()),
            "MRR": float((1 / pos).mean())}


def recuperadores():
    docs, perguntas = documentos_ficticios(), perguntas_ficticias()
    frags = indexar(docs)
    configs = {"TF-IDF\npalavras": {}, "TF-IDF\npalavras 1-2": {"ngram_range": (1, 2)},
               "TF-IDF\ncaracteres 3-5": {"analyzer": "char_wb", "ngram_range": (3, 5)}}
    res = {nome: avaliar(RecuperadorTFIDF(frags, **op), perguntas) for nome, op in configs.items()}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    largura = 0.25
    for k, (metrica, cor) in enumerate(zip(["hit@1", "hit@3", "MRR"], [AZUL, LARANJA, VERDE])):
        valores = [res[n][metrica] for n in configs]
        barras = ax.bar(np.arange(3) + (k - 1) * largura, valores, largura, label=metrica, color=cor)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), list(configs))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("métrica de recuperação")
    ax.set_title("Comparação de recuperadores (24 perguntas fictícias)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, "fig_m9_recuperadores.png")


def chunking():
    docs, perguntas = documentos_ficticios(), perguntas_ficticias()
    tamanhos = [10, 20, 40, 80, 160]
    res = [avaliar(RecuperadorTFIDF(indexar(docs, t, t // 4)), perguntas) for t in tamanhos]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for metrica, cor, marc in [("hit@1", AZUL, "o"), ("MRR", LARANJA, "s")]:
        ax.plot(range(len(tamanhos)), [r[metrica] for r in res], marker=marc, color=cor, label=metrica)
    ax.set_xticks(range(len(tamanhos)), [str(t) for t in tamanhos])
    ax.set_xlabel("tamanho do fragmento (palavras; sobreposição de 25%)")
    ax.set_ylabel("métrica")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Efeito do tamanho do fragmento na recuperação", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    salvar(fig, "fig_m9_chunking.png")


def scores():
    rec = RecuperadorTFIDF(indexar(documentos_ficticios()))
    ok = [rec.buscar(p, 1)[0][0] for p, _ in perguntas_ficticias()]
    fora = [rec.buscar(p, 1)[0][0] for p in perguntas_fora_da_base()]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.scatter(ok, 1 + rng.uniform(-0.12, 0.12, len(ok)), color=AZUL, alpha=0.8, label="com resposta na base")
    ax.scatter(fora, 0 + rng.uniform(-0.12, 0.12, len(fora)), color=LARANJA, alpha=0.9, label="sem resposta na base")
    for lim, est in [(0.12, ":"), (0.25, "--")]:
        ax.axvline(lim, color=CINZA, linestyle=est)
        ax.text(lim + 0.005, 1.4, f"limiar {lim:.2f}", fontsize=8, color=CINZA)
    ax.set_yticks([0, 1], ["sem resposta", "com resposta"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("similaridade do cosseno do melhor fragmento")
    ax.set_title("Os escores se sobrepõem: nenhum limiar separa tudo", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    salvar(fig, "fig_m9_scores.png")


if __name__ == "__main__":
    heatmap_atencao()
    posicional()
    recuperadores()
    chunking()
    scores()
    print("figuras salvas em", os.path.abspath(OUT))

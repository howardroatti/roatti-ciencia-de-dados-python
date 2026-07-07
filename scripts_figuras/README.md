# Scripts de figuras

Estes scripts **regeneram as figuras de resultado do livro** a partir de execuções
reais do scikit-learn sobre conjuntos de dados públicos — garantindo que toda figura
do livro seja reproduzível.

| Script | Gera |
|---|---|
| `figuras_resultado_m6.py` | Fronteiras de decisão (kNN, SVM, árvore, Random Forest, regressão logística, Perceptron, MLP), curvas de erro, importância de atributos, PCA, agrupamentos e regressão — Módulo VI |

Execute:

```bash
python scripts_figuras/figuras_resultado_m6.py
```

As imagens são salvas em `scripts_figuras/figuras/` (ou na pasta de figuras do livro,
quando executado dentro do projeto do livro).

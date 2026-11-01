# Scripts de figuras

Estes scripts **regeneram as figuras de resultado do livro** a partir de execuções
reais do código do livro sobre dados fictícios gerados por código ou conjuntos embutidos do scikit-learn — garantindo que toda figura
do livro seja reproduzível.

| Script | Gera |
|---|---|
| `figuras_resultado_m6.py` | Fronteiras de decisão (kNN, SVM, árvore, Random Forest, regressão logística, Perceptron, MLP), curvas de erro, importância de atributos, PCA, agrupamentos e regressão — Módulo VI |
| `figuras_dados_sinteticos_m6.py` | Importância de atributos pelo Random Forest e listas de seleção (Módulo VI), sobre os imóveis **fictícios** de `src/dados_sinteticos_m6.py` |
| `figuras_m3_m5.py` | Boxplot e mapa de correlação sobre dados fictícios (Módulo V) |
| `figuras_m2_m3_expansao.py` | Broadcasting (Módulo II) e série temporal com média móvel (Módulo III) |
| `figuras_m5_expansao.py` | Eixo truncado, paletas, distribuições, relação, série temporal e pequenos múltiplos (Módulo V) |
| `figuras_m6_xgboost.py` | Amostra da busca em grade do XGBoost (Módulo VI) |
| `figuras_m7_expansao.py` | Dígitos, *overfitting*, otimizadores, MLP × CNN, LSTM e autoencoder (Módulo VII) |
| `figuras_m8_m9_expansao.py` | Atenção, codificação posicional, recuperadores, *chunking* e escores do RAG (Módulos VIII e IX) |
| `figuras_m10_m11_expansao.py` | Laço ReAct e métricas do *harness* (Módulos X e XI) |

Execute:

```bash
python scripts_figuras/figuras_resultado_m6.py
```

As imagens são salvas em `scripts_figuras/figuras/` (ou na pasta de figuras do livro,
quando executado dentro do projeto do livro).

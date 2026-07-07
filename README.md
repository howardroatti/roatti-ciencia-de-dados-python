# Ciência de Dados com Python — Código Companheiro

Código-fonte, notebooks e scripts que acompanham o livro
**_Ciência de Dados com Python: dos Fundamentos aos Agentes de IA Generativa_**,
de **Howard Roatti**.

Todo o material aqui é **reproduzível**: os exemplos dos Módulos I a VI executam de
ponta a ponta sobre conjuntos de dados públicos, e as figuras de resultado do livro
são geradas por estes scripts.

## Estrutura

```
codigo/
├── notebooks/          Um notebook por módulo (código do livro, executável)
├── scripts_figuras/    Scripts que regeneram as figuras de resultado do livro
├── data/               Conjuntos de dados de exemplo (+ instruções para os externos)
├── src/                Utilidades compartilhadas
├── requirements.txt    Dependências do núcleo (Módulos I–VI)
├── requirements-extra.txt  Dependências opcionais (boosting, deploy, LLMs)
├── pyproject.toml
└── LICENSE             MIT
```

## Instalação

Requer **Python 3.10+** (testado em 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para os capítulos avançados (Módulo VI com XGBoost/deploy e Módulos VIII–XI com
LLMs), instale conforme a necessidade:

```bash
pip install -r requirements-extra.txt
```

## Como usar

- **Notebooks** (`notebooks/`): abra com Jupyter (`jupyter lab`) e execute célula a
  célula. Cada notebook corresponde a um módulo do livro.
- **Figuras** (`scripts_figuras/`): para regenerar as figuras de resultado, execute,
  por exemplo:

  ```bash
  python scripts_figuras/figuras_resultado_m6.py
  ```

## Conjuntos de dados

Os exemplos usam principalmente conjuntos **embutidos no scikit-learn**
(`load_iris`, `load_breast_cancer`, `load_diabetes`) e alguns arquivos de exemplo em
`data/`. Bases externas (Kaggle) que não podem ser redistribuídas têm instruções de
download em [`data/README.md`](data/README.md).

## Mapa de módulos

| Módulo | Tema | Executa localmente? |
|---|---|---|
| I–V | Python, NumPy, Pandas, visualização | ✅ Sim |
| VI | Machine Learning (scikit-learn) | ✅ Sim (XGBoost opcional) |
| VII | Deep Learning (conceitual) | — teórico |
| VIII | Transformers e LLMs | ⚙️ Requer `torch`/`transformers` |
| IX | IA Generativa e RAG | ⚙️ Requer modelos locais (Ollama/HF) |
| X | Agentes inteligentes | ⚙️ Requer LLM local |
| XI | IA responsável e produção | ⚙️ Parcial |

## Licença

Distribuído sob a licença **MIT** — veja [`LICENSE`](LICENSE).

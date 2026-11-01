# Ciência de Dados com Python — Código Companheiro

Código-fonte, notebooks e scripts que acompanham o livro
**_Ciência de Dados com Python: dos Fundamentos aos Agentes de IA Generativa_**,
de **Howard Roatti**.

Todo o material aqui é **reproduzível**: os exemplos executam de
ponta a ponta com dados fictícios gerados por código ou conjuntos embutidos do scikit-learn, e as figuras de resultado do livro
são geradas por estes scripts.

## Estrutura

```
codigo/
├── notebooks/          Um notebook por módulo (código do livro, executável)
├── scripts_figuras/    Scripts que regeneram as figuras de resultado do livro
├── data/               Conjuntos de dados de exemplo (+ instruções para os externos)
├── src/                Geradores de dados fictícios (dados_sinteticos_*.py)
├── requirements.txt    Dependências do núcleo (Módulos I–VI e Anexo A)
├── requirements-extra.txt  Dependências opcionais (boosting, torch, agentes, LLMs)
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

Para os capítulos que usam pacotes adicionais (XGBoost e Flask no VI, `torch` no VII,
`fastapi`/`pydantic` nos X e XI, modelos reais nos IX–XI), instale conforme a necessidade:

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
`data/`. Os exemplos usam dados fictícios gerados por código (veja `data/README.md`); bases externas, quando citadas, têm instruções de
download em [`data/README.md`](data/README.md).

## Mapa de módulos

Cada módulo do livro tem um notebook com o **código** (as explicações estão no livro). Todos
os exemplos usam dados fictícios gerados por código (`src/dados_sinteticos_*.py`) ou
conjuntos embutidos do scikit-learn.

| Módulo | Notebook | Executa localmente? |
|---|---|---|
| I Python | `notebooks/01-modulo-i.ipynb` | ✅ Sim (web scraping com cópia local, sem rede) |
| II NumPy | `notebooks/02-modulo-ii.ipynb` | ✅ Sim |
| III Pandas | `notebooks/03-modulo-iii.ipynb` | ✅ Sim |
| IV Aquisição de dados | `notebooks/04-modulo-iv.ipynb` | ✅ Sim (cria arquivos de exemplo) |
| V Visualização | `notebooks/05-modulo-v.ipynb` | ✅ Sim |
| VI Machine Learning | `notebooks/06-modulo-vi.ipynb` | ✅ Sim (XGBoost e Flask opcionais) |
| VII Deep Learning | `notebooks/07-modulo-vii.ipynb` | ✅ Sim, em CPU (requer `torch`) |
| VIII Transformers e LLMs | `notebooks/08-modulo-viii.ipynb` | ✅ Sim, mecanismo em NumPy |
| IX IA Generativa e RAG | `notebooks/09-modulo-ix.ipynb` | ✅ RAG offline; células com modelos reais são opcionais |
| X Agentes | `notebooks/10-modulo-x.ipynb` | ✅ Agente do zero offline; `ollama` opcional |
| XI IA responsável e LLMOps | `notebooks/11-modulo-xi.ipynb` | ✅ Laboratório offline (testes com `TestClient`) |
| Anexo A | `notebooks/90-anexo-a-resolucoes.ipynb` | ✅ Sim (célula de preparação cria dados de exemplo) |

As células marcadas com `# OPCIONAL: requer ...` dependem de pacotes ou modelos que não
fazem parte do núcleo (veja `requirements-extra.txt`).

## Licença

Distribuído sob a licença **MIT** — veja [`LICENSE`](LICENSE).

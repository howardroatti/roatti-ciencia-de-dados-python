# Conjuntos de dados

Esta pasta reúne os dados usados nos exemplos do livro.

## Incluídos no repositório

| Arquivo | Descrição | Origem |
|---|---|---|
| `DataSets/iris.csv` | Flores Iris (150 amostras, 4 atributos + espécie) | Gerado do `sklearn.datasets.load_iris` (domínio público) |
| `exemplo1.csv` | Exemplo didático de leitura sem cabeçalho (Módulo IV) | Sintético |
| `exemplo2.csv` | Exemplo didático separado por tabulação (Módulo IV) | Sintético |
| `exemplo3.csv` | Exemplo didático de `skiprows`/`nrows` (Módulo IV) | Sintético |

Vários exemplos usam **conjuntos embutidos do scikit-learn** — não precisam de
arquivo: `load_iris()`, `load_breast_cancer()`, `load_diabetes()`.

## Não incluídos (baixe conforme a licença de cada um)

Alguns exemplos citam bases que **não podem ser redistribuídas** aqui por questões
de licença. Baixe-as na fonte e coloque em `data/` (ou `data/kaggle/`):

| Arquivo esperado | Onde obter |
|---|---|
| `DataSets/titanic_treino.csv` | Kaggle — *Titanic: Machine Learning from Disaster* (`train.csv`) |
| `DataSets/Clientes_Cartao_Credito.csv` | Kaggle — *Default of Credit Card Clients* |
| `DataSets/house_sales.csv` | Kaggle — *House Sales* |
| `Melbourne-House-Snapshot/melb_data.csv` | Kaggle — *Melbourne Housing Snapshot* |
| `loan_data.csv.gz` | Kaggle — *Loan Data* |

Onde a base específica não é essencial (o exemplo ilustra uma **técnica**, não os
dados), você pode substituí-la por `load_iris`/`load_breast_cancer`/`load_diabetes`
sem prejuízo pedagógico.

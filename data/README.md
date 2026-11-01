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

## Dados fictícios gerados por código

Os exemplos dos Módulos III, V e VI **não dependem de bases externas**: usam conjuntos
**fictícios**, gerados por código com semente fixa (resultados reproduzíveis), em
`src/dados_sinteticos_m3m5.py` e `src/dados_sinteticos_m6.py`. Eles imitam a estrutura
de bases conhecidas (passageiros, clientes de cartão, vendas de casas, imóveis,
empréstimos), mas **nenhum dado real é redistribuído** e não há restrição de licença.

Se quiser praticar com bases reais, baixe-as na fonte e **verifique a licença de cada
uma** antes de qualquer uso além do estudo pessoal.

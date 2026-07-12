# Checklist de Atendimento ao Enunciado

Este arquivo mapeia os itens pedidos no trabalho para os arquivos entregues no repositorio.

## Interpretacao confirmada pelo professor

- Nao sao 5 objetos no total.
- Sao 5 classes de objetos.
- Cada classe aparece 5 vezes.
- Portanto, cada cena possui 25 objetos: 5 circulos, 5 quadrados, 5 elipses, 5 retangulos e 5 triangulos.
- O objetivo principal nao e apenas classificar, mas testar reasoning.
- O experimento treina em uma cena balanceada e testa em datasets/cenas aleatorios distintos.

## Estrutura de dados

| Item do enunciado | Status | Onde esta |
|---|---|---|
| CLEVR simplificado com vetor de 11 atributos | Atendido | `clevr_ltn_experimentos.py`, secao 2 do relatorio |
| Coordenadas normalizadas `(x,y)` | Atendido | `generate_scene` em `clevr_ltn_experimentos.py` |
| Cores one-hot: vermelho, verde, azul | Atendido | `COLORS`, `generate_scene` |
| Formas one-hot: circulo, quadrado, elipse, retangulo, triangulo | Atendido | `SHAPES`, `generate_scene` |
| 5 objetos de cada classe de forma | Atendido | `balanced_shape_indices`, colunas `n_circle` a `n_triangle` no CSV |
| 25 objetos por cena | Atendido | parametro padrao `--n-objects 25` |
| Plotar cenario aleatorio | Atendido | pasta `figuras/` |

## Protocolo experimental

| Item | Status | Onde esta |
|---|---|---|
| Treinar reasoning com uma unica cena/caso | Atendido | `train_seed=2025` em `run(args)` |
| Testar em datasets aleatorios distintos | Atendido | seeds de teste 42, 43, 44, 45 e 46 |
| Manter 25 objetos em cada dataset de teste | Atendido | `generate_scene(seed, 25)` |
| Manter 5 objetos de cada classe em cada dataset de teste | Atendido | `balanced_shape_indices` e contagens no CSV |

## Tarefa 1 - Taxonomia e formas

| Item | Status | Onde esta |
|---|---|---|
| `isEllipse(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isRectangle(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isTriangle(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isCircle(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isSquare(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isSmall(x)` e `isBig(x)` | Atendido | `Model.size`, relatorio secao 3 |
| Forma unica | Atendido | `shape_unique` no CSV |
| Cobertura/completude de formas | Atendido | `shape_coverage` no CSV |

## Tarefa 2 - Raciocinio espacial horizontal

| Item | Status | Onde esta |
|---|---|---|
| `LeftOf(x,y)` | Atendido | `gt_left`, `Model.rel`, CSV |
| `RightOf(x,y)` | Atendido | `gt_right`, `Model.rel`, CSV |
| `CloseTo(x,y)` | Atendido | `gt_close_soft`, `Model.rel`, CSV |
| `InBetween(x,y,z)` | Atendido | `gt_between`, `Model.between`, CSV |
| Irreflexividade de `LeftOf` | Atendido | `left_irreflexive` |
| Assimetria de `LeftOf` | Atendido | `left_asymmetric` |
| Inverso `LeftOf/RightOf` | Atendido | `left_right_inverse` |
| Transitividade horizontal | Atendido | `left_transitive` |
| Objeto mais a esquerda | Atendido | `last_left` |
| Objeto mais a direita | Atendido | `last_right` |

## Tarefa 3 - Raciocinio vertical

| Item | Status | Onde esta |
|---|---|---|
| `Below(x,y)` | Atendido | `gt_below`, `Model.rel`, CSV |
| `Above(x,y)` | Atendido | `gt_above`, `Model.rel`, CSV |
| Inverso `Below/Above` | Atendido | `below_above_inverse` |
| Transitividade vertical | Atendido | `below_transitive` |
| `CanStack(x,y)` | Atendido | `gt_can_stack`, `can_stack_rule` |
| Restricao de suporte nao ser retangulo nem triangulo | Atendido | `gt_can_stack`, `can_stack_rule` |
| Estabilidade por mesmo tamanho ou alinhamento horizontal | Atendido | `gt_can_stack` |

## Tarefa 4 - Raciocinio composto

| Consulta | Status | Onde esta |
|---|---|---|
| Objeto pequeno abaixo de elipse e a esquerda de quadrado | Atendido | `query_small_below_ellipse_left_square` |
| Retangulo verde entre dois objetos | Atendido | `query_green_rectangle_between` |
| Triangulos proximos devem ter mesmo tamanho | Atendido | `query_triangles_close_same_size` |

## Entregas

| Item pedido | Status | Onde esta |
|---|---|---|
| Codigo no GitHub | Atendido | `clevr_ltn_experimentos.py` |
| Texto em Markdown/PDF | Atendido | `RELATORIO_TRABALHO_LTN.md` e PDF local gerado |
| Breve descricao de NeSy e LTN | Atendido | secao 1 do relatorio |
| Descricao do dataset CLEVR simplificado | Atendido | secao 2 do relatorio |
| Valor de satisfacao das formulas | Atendido | `resultados_clevr_ltn.csv` e secao 7 do relatorio |
| 5 execucoes em datasets aleatorios distintos | Atendido | seeds de teste 42, 43, 44, 45, 46 |
| `satAgg` de cada pergunta/formula | Atendido | CSV e tabela de satisfatibilidade no relatorio |
| Interpretabilidade das consultas existenciais | Atendido | colunas `q1_*`, `q2_*` e secao 8 do relatorio |
| Acuracia | Atendido | CSV e relatorio |
| Precisao | Atendido | CSV e relatorio |
| Recall | Atendido | CSV e relatorio |
| F1 Score | Atendido | CSV e relatorio |
| Explicacao das perguntas para ponto extra | Atendido | secao 8 do relatorio |

## Arquivos principais para correcao

- `README.md`: guia de execucao e links.
- `clevr_ltn_experimentos.py`: implementacao principal.
- `trabalho_ltn_clevr.ipynb`: notebook executavel.
- `RELATORIO_TRABALHO_LTN.md`: relatorio textual.
- `resultados_clevr_ltn.csv`: resultados completos.
- `figuras/`: cenas geradas.
- `.github/workflows/test.yml`: teste automatico rapido.
- `.github/workflows/full-experiment.yml`: experimento completo automatico.

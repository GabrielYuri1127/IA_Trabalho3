# Checklist de Atendimento ao Enunciado

Este arquivo mapeia os itens pedidos no trabalho para os arquivos entregues no repositorio.

## Estrutura de dados

| Item do enunciado | Status | Onde esta |
|---|---|---|
| CLEVR simplificado com vetor de 11 atributos | Atendido | `clevr_ltn_experimentos.py`, secao 2 do relatorio |
| Coordenadas normalizadas `(x,y)` | Atendido | `generate_scene` em `clevr_ltn_experimentos.py` |
| Cores one-hot: vermelho, verde, azul | Atendido | `COLORS`, `generate_scene` |
| Formas one-hot: circulo, quadrado, cilindro, cone, triangulo | Atendido | `SHAPES`, `generate_scene` |
| Tamanho pequeno/grande | Atendido | atributo `[10]`, predicados `is_small` e `is_big` |
| 25 objetos aleatorios por cena | Atendido | parametro padrao `--n-objects 25` |
| Plotar cenario aleatorio | Atendido | pasta `figuras/` |

## Tarefa 1 - Taxonomia e formas

| Item | Status | Onde esta |
|---|---|---|
| `isCylinder(x)` | Atendido | `Model.shape`, relatorio secao 3 |
| `isCone(x)` | Atendido | `Model.shape`, relatorio secao 3 |
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
| Restricao de suporte nao ser cone nem triangulo | Atendido | `gt_can_stack`, `can_stack_rule` |
| Estabilidade por mesmo tamanho ou alinhamento horizontal | Atendido | `gt_can_stack` |

## Tarefa 4 - Raciocinio composto

| Consulta | Status | Onde esta |
|---|---|---|
| Objeto pequeno abaixo de cilindro e a esquerda de quadrado | Atendido | `query_small_below_cylinder_left_square` |
| Cone verde entre dois objetos | Atendido | `query_green_cone_between` |
| Triangulos proximos devem ter mesmo tamanho | Atendido | `query_triangles_close_same_size` |

## Entregas

| Item pedido | Status | Onde esta |
|---|---|---|
| Codigo no GitHub | Atendido | `clevr_ltn_experimentos.py` |
| Texto em Markdown/PDF | Atendido | `RELATORIO_TRABALHO_LTN.md` e PDF local gerado |
| Breve descricao de NeSy e LTN | Atendido | secao 1 do relatorio |
| Descricao do dataset CLEVR simplificado | Atendido | secao 2 do relatorio |
| Valor de satisfacao das formulas | Atendido | `resultados_clevr_ltn.csv` e secao 7 do relatorio |
| 5 execucoes com datasets aleatorios distintos | Atendido | seeds 42, 43, 44, 45, 46 |
| `satAgg` de cada pergunta/formula | Atendido | CSV e tabela de satisfatibilidade no relatorio |
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

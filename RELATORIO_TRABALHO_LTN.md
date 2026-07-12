# Raciocinio Espacial Neuro-Simbolico com LTNtorch

Disciplina: Fundamentos de Inteligencia Artificial  
Tema: Logic Tensor Networks (LTN), CLEVR simplificado e raciocinio espacial  

## 1. Introducao: IA Neuro-Simbolica e LTN

A Inteligencia Artificial neuro-simbolica (NeSy) combina redes neurais, que aprendem padroes a partir de dados, com representacoes simbolicas, que permitem expressar conhecimento por regras logicas. Em vez de treinar apenas um modelo do tipo entrada -> saida, uma abordagem NeSy permite que o treinamento seja guiado por conhecimento declarativo, como "se um objeto esta a esquerda de outro, entao o segundo nao esta a esquerda do primeiro".

Logic Tensor Networks (LTN) sao uma familia de modelos neuro-simbolicos que mapeiam elementos da logica de primeira ordem para tensores diferenciaveis. Predicados como `LeftOf(x,y)` ou `IsSquare(x)` sao implementados por redes neurais ou funcoes diferenciaveis que retornam valores de verdade fuzzy no intervalo `[0,1]`. As formulas logicas sao avaliadas com operadores fuzzy, e a satisfatibilidade da base de conhecimento e usada como funcao objetivo durante o treinamento.

No LTNtorch, esse processo segue cinco etapas principais: aterramento dos dados nas formulas, avaliacao dos valores de verdade, agregacao da satisfatibilidade, calculo da perda `1 - satAgg` e retropropagacao para ajustar os parametros dos predicados.

Neste trabalho, o grounding usado foi:

| Elemento logico | Implementacao no experimento |
|---|---|
| Constantes/objetos | Vetores reais de 11 atributos |
| Variaveis `x`, `y`, `z` | Conjuntos de objetos da cena |
| Predicados unarios | Redes neurais para cor, forma e tamanho |
| Predicados binarios | Redes neurais para relacoes espaciais entre pares |
| Predicado ternario | Rede neural para `InBetween(x,y,z)` |
| Formulas logicas | Operadores fuzzy diferenciaveis e agregacao `satAgg` |

Os conectivos fuzzy foram implementados com negacao padrao `1 - p`, conjuncao por produto, disjuncao probabilistica, implicacao de Reichenbach e agregadores de media-p para `forall` e `exists`. Alem disso, foram usados fatos supervisionados gerados a partir da geometria da cena. Isso evita uma solucao degenerada em que o modelo satisfaz axiomas estruturais de forma trivial, por exemplo aprendendo `LeftOf(x,y)=0` para todos os pares apenas para satisfazer assimetria.

## 2. Dataset CLEVR Simplificado

O trabalho usa uma versao simplificada do CLEVR. Em vez de processar imagens reais, cada objeto e representado por um vetor de 11 atributos:

| Posicoes | Significado |
|---|---|
| `[0,1]` | Coordenadas normalizadas `(x,y)` |
| `[2,3,4]` | Cor one-hot: vermelho, verde, azul |
| `[5,6,7,8,9]` | Forma one-hot: circulo, quadrado, elipse, retangulo, triangulo |
| `[10]` | Tamanho: pequeno `0.0`, grande `1.0` |

Conforme a orientacao do professor, cada cena possui 5 classes de objetos, com 5 objetos de cada classe de forma, totalizando 25 objetos por cena:

| Classe de forma | Quantidade por cena |
|---|---:|
| Circulo | 5 |
| Quadrado | 5 |
| Elipse | 5 |
| Retangulo | 5 |
| Triangulo | 5 |

O experimento treina o reasoning em uma unica cena balanceada (`train_seed=2025`) e testa a generalizacao em 5 cenas aleatorias distintas (`seed=42` a `seed=46`).

Para evitar sobreposicao visual forte entre objetos, a geracao das posicoes usa rejeicao amostral com distancia minima de `0.08` entre centroides. O CSV registra `min_pair_distance` e `overlap_ok` para auditar esse tratamento.

## 3. Predicados Implementados

Predicados unarios:

- `IsCircle(x)`, `IsSquare(x)`, `IsEllipse(x)`, `IsRectangle(x)`, `IsTriangle(x)`
- `IsSmall(x)`, `IsBig(x)`
- `IsRed(x)`, `IsGreen(x)`, `IsBlue(x)`

Predicados binarios:

- `LeftOf(x,y)`, `RightOf(x,y)`
- `Below(x,y)`, `Above(x,y)`
- `CloseTo(x,y)`
- `SameSize(x,y)`
- `CanStack(x,y)`

Predicado ternario:

- `InBetween(x,y,z)`, indicando que `x` esta entre `y` e `z` no eixo horizontal.

## 4. Base de Conhecimento

As formulas principais usadas para treinar e auditar o modelo foram:

1. Forma unica:

```text
forall x: not(IsShape_i(x) and IsShape_j(x)), para i != j
```

2. Cobertura de formas:

```text
forall x: IsCircle(x) or IsSquare(x) or IsEllipse(x) or IsRectangle(x) or IsTriangle(x)
```

3. Irreflexividade de esquerda:

```text
forall x: not(LeftOf(x,x))
```

4. Assimetria:

```text
forall x,y: LeftOf(x,y) -> not(LeftOf(y,x))
```

5. Inverso esquerda-direita:

```text
forall x,y: LeftOf(x,y) <-> RightOf(y,x)
```

6. Transitividade horizontal:

```text
forall x,y,z: (LeftOf(x,y) and LeftOf(y,z)) -> LeftOf(x,z)
```

7. Objeto mais a esquerda:

```text
exists x: forall y, LeftOf(x,y)
```

8. Objeto mais a direita:

```text
exists x: forall y, RightOf(x,y)
```

9. Inverso abaixo-acima:

```text
forall x,y: Below(x,y) <-> Above(y,x)
```

10. Transitividade vertical:

```text
forall x,y,z: (Below(x,y) and Below(y,z)) -> Below(x,z)
```

11. Objeto entre dois outros:

```text
InBetween(x,y,z) <-> (LeftOf(y,x) and RightOf(z,x)) or (LeftOf(z,x) and RightOf(y,x))
```

12. Regra de empilhamento:

```text
CanStack(x,y) -> not(IsRectangle(y) or IsTriangle(y))
```

No treinamento supervisionado dos predicados, `CanStack(x,y)` tambem considera a estabilidade pedida no enunciado: o suporte `y` nao pode ser retangulo nem triangulo e os objetos devem ter o mesmo tamanho ou centroide horizontal suficientemente alinhado.

## 5. Consultas Compostas

Consulta 1: existe algum objeto pequeno abaixo de uma elipse e a esquerda de um quadrado?

```text
exists x: IsSmall(x) and exists y: IsEllipse(y) and Below(x,y)
          and exists z: IsSquare(z) and LeftOf(x,z)
```

Consulta 2: existe um retangulo verde entre dois objetos quaisquer?

```text
exists x,y,z: IsRectangle(x) and IsGreen(x) and InBetween(x,y,z)
```

Consulta 3: se dois triangulos estao proximos, entao possuem o mesmo tamanho?

```text
forall x,y: (IsTriangle(x) and IsTriangle(y) and CloseTo(x,y)) -> SameSize(x,y)
```

Relacao com os niveis de raciocinio do enunciado:

| Consulta | Nivel | Por que atende |
|---|---|---|
| q1 | Nivel 2 - filtragem relacional composta | Combina tamanho, forma, relacao vertical e relacao horizontal |
| q2 | Nivel 2 - deducao de posicao absoluta | Usa `InBetween` para decidir se o retangulo verde esta entre objetos |
| q3 | Nivel 4 - implicacao material | Avalia uma regra condicional universal e evidencia o caso de verdade vacua quando nao ha pares proximos |

## 6. Metricas

As metricas foram calculadas a partir da matriz de confusao:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 * Precision * Recall / (Precision + Recall)
```

Tambem foi reportado o valor de satisfatibilidade global `satAgg`, que indica o quanto a base de conhecimento esta sendo satisfeita pelo modelo.

As metricas foram registradas em tres granularidades no CSV: `unary_*` para predicados de cor/forma/tamanho, `relation_*` para relacoes espaciais e `all_*` para o conjunto completo. As colunas principais `accuracy`, `precision`, `recall` e `f1` preservam a avaliacao das relacoes espaciais, que sao a parte mais dificil do reasoning.

## 7. Resultados

O experimento completo foi executado com uma cena de treino balanceada (`train_seed=2025`) e 5 cenas aleatorias de teste, todas com 25 objetos e 5 objetos por classe de forma. O arquivo completo gerado esta em `resultados_clevr_ltn.csv`.

Comando usado:

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 \
  --train-seed 2025 --seed 42 --min-distance 0.08 \
  --out resultados_clevr_ltn.csv --plot-dir figuras
```

| Execucao | Train seed | Test seed | satAgg | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2025 | 42 | 0.6007 | 0.8590 | 0.7440 | 0.7613 | 0.7526 |
| 2 | 2025 | 43 | 0.5987 | 0.7964 | 0.6287 | 0.7411 | 0.6803 |
| 3 | 2025 | 44 | 0.5941 | 0.8020 | 0.6448 | 0.7411 | 0.6896 |
| 4 | 2025 | 45 | 0.5905 | 0.8060 | 0.6602 | 0.6994 | 0.6792 |
| 5 | 2025 | 46 | 0.6050 | 0.8178 | 0.6796 | 0.7521 | 0.7140 |

Resumo:

| Metrica | Media | Desvio padrao |
|---|---:|---:|
| satAgg | 0.5978 | 0.0051 |
| Accuracy relacional | 0.8163 | 0.0225 |
| Precision relacional | 0.6715 | 0.0400 |
| Recall relacional | 0.7390 | 0.0212 |
| F1 relacional | 0.7031 | 0.0277 |
| Accuracy unaria | 0.9808 | 0.0148 |
| F1 unario | 0.9667 | 0.0261 |
| Accuracy geral | 0.8183 | 0.0223 |
| F1 geral | 0.7062 | 0.0277 |

Contagem de classes nas cenas de teste:

| Execucao | Circulos | Quadrados | Elipses | Retangulos | Triangulos |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 5 | 5 |
| 2 | 5 | 5 | 5 | 5 | 5 |
| 3 | 5 | 5 | 5 | 5 | 5 |
| 4 | 5 | 5 | 5 | 5 | 5 |
| 5 | 5 | 5 | 5 | 5 | 5 |

Tratamento de overlapping:

| Execucao | Test seed | Distancia minima entre centroides | `overlap_ok` |
|---:|---:|---:|---:|
| 1 | 42 | 0.0807 | 1 |
| 2 | 43 | 0.0842 | 1 |
| 3 | 44 | 0.0805 | 1 |
| 4 | 45 | 0.0816 | 1 |
| 5 | 46 | 0.0810 | 1 |

Satisfatibilidade media das formulas e perguntas:

| Coluna no CSV | Formula/pergunta | Media | Desvio padrao |
|---|---|---:|---:|
| `shape_unique` | Forma unica | 0.9939 | 0.0043 |
| `shape_coverage` | Cobertura de formas | 0.7333 | 0.0703 |
| `left_irreflexive` | LeftOf irreflexivo | 0.9292 | 0.0155 |
| `left_asymmetric` | LeftOf assimetrico | 0.9371 | 0.0099 |
| `left_right_inverse` | Left/Right inversos | 0.8396 | 0.0096 |
| `left_transitive` | LeftOf transitivo | 0.9786 | 0.0046 |
| `below_above_inverse` | Below/Above inversos | 0.8802 | 0.0075 |
| `below_transitive` | Below transitivo | 0.9729 | 0.0080 |
| `between_rule` | Regra InBetween | 0.6558 | 0.0275 |
| `last_left` | Objeto mais a esquerda | 0.4182 | 0.0052 |
| `last_right` | Objeto mais a direita | 0.4409 | 0.0121 |
| `can_stack_rule` | Regra CanStack - suporte valido | 0.9792 | 0.0167 |
| `can_stack_stability_rule` | Regra CanStack - estabilidade | 0.7802 | 0.0374 |
| `query_small_below_ellipse_left_square` | Pergunta composta 1 | 0.0506 | 0.0119 |
| `query_green_rectangle_between` | Pergunta composta 2 | 0.1416 | 0.0429 |
| `query_triangles_close_same_size` | Pergunta composta 3 | 0.9785 | 0.0134 |
| `ltn_api_sat_check` | Auditoria complementar via LTNtorch | 0.9110 | 0.0169 |

Evidencia interpretavel das consultas compostas:

| Execucao | Test seed | q1 GT | q1 melhor testemunha | q1 objeto | q2 GT | q2 melhor testemunha | q2 objeto | q3 pares proximos | q3 violacoes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 1 | 0.9989 | 20 | 1 | 0.9885 | 0 | 2 | 2 |
| 2 | 43 | 1 | 0.9950 | 6 | 1 | 0.9983 | 19 | 2 | 2 |
| 3 | 44 | 1 | 0.9985 | 7 | 1 | 0.9925 | 18 | 2 | 0 |
| 4 | 45 | 1 | 0.9983 | 11 | 1 | 0.9994 | 2 | 4 | 0 |
| 5 | 46 | 1 | 0.9949 | 8 | 1 | 0.9994 | 9 | 6 | 4 |

Auditoria XAI do par horizontal extremo:

| Execucao | Test seed | Objeto mais a esquerda | Objeto mais a direita | `LeftOf(esq,dir)` | `LeftOf(dir,esq)` | Assimetria |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 2 | 10 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 43 | 1 | 12 | 1.0000 | 0.0000 | 1.0000 |
| 3 | 44 | 10 | 15 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 45 | 9 | 19 | 1.0000 | 0.0000 | 1.0000 |
| 5 | 46 | 10 | 24 | 1.0000 | 0.0000 | 1.0000 |

`q1 GT` e `q2 GT` indicam a resposta booleana calculada diretamente dos atributos e relacoes geometricas da cena. A coluna de melhor testemunha mostra a maior confianca fuzzy encontrada pelo modelo para algum objeto que satisfaz a consulta. Isso ajuda a interpretar casos em que o `satAgg` existencial e baixo: a agregacao fuzzy considera muitos objetos que nao satisfazem a pergunta, mas a melhor testemunha pode confirmar que a consulta foi encontrada.

A queda natural nas metricas em relacao a um treino feito separadamente em cada cena e esperada: agora o modelo aprende em uma unica cena e e avaliado em cenas aleatorias novas. Isso deixa o experimento mais alinhado com a ideia do professor de testar generalizacao de regras de reasoning com poucos dados.

Figuras geradas para visualizacao das cenas:

| Papel | Seed | Figura |
|---|---:|---|
| Treino | 2025 | [scene_seed_2025.png](figuras/scene_seed_2025.png) |
| Teste | 42 | [scene_seed_42.png](figuras/scene_seed_42.png) |
| Teste | 43 | [scene_seed_43.png](figuras/scene_seed_43.png) |
| Teste | 44 | [scene_seed_44.png](figuras/scene_seed_44.png) |
| Teste | 45 | [scene_seed_45.png](figuras/scene_seed_45.png) |
| Teste | 46 | [scene_seed_46.png](figuras/scene_seed_46.png) |

## 8. Explicacao do Raciocinio das Perguntas

Pergunta composta 1: o modelo procura um objeto `x` pequeno. Depois verifica se existe um objeto `y` que seja elipse e esteja acima de `x`, isto e, `Below(x,y)`. Por fim, procura um objeto `z` que seja quadrado e esteja a direita de `x`, isto e, `LeftOf(x,z)`. A satisfatibilidade depende de as tres condicoes aparecerem juntas na mesma cena.

Pergunta composta 2: o modelo procura um objeto `x` que seja simultaneamente retangulo e verde. Em seguida, usa `InBetween(x,y,z)` para verificar se esse retangulo fica entre outros dois objetos no eixo horizontal. Como a cena e aleatoria, a consulta pode ter baixa satisfatibilidade quando nao ha retangulo verde em posicao intermediaria.

Pergunta composta 3: o modelo avalia todos os pares de objetos. Quando dois objetos sao triangulos e estao proximos segundo `CloseTo(x,y)`, a regra exige que `SameSize(x,y)` tambem seja verdadeiro. Esta e uma implicacao universal; portanto, ela testa consistencia global da cena, nao apenas a existencia de um exemplo.

Para as consultas existenciais, o CSV tambem registra a melhor testemunha. Em q1 e q2, todas as cinco cenas tiveram ground-truth positivo (`q1_gt_exists=1` e `q2_gt_exists=1`) e confianca maxima proxima de 1.0. Isso torna a resposta mais explicavel do que olhar apenas para o `satAgg` agregado, pois a agregacao existencial pode ficar baixa mesmo quando ha uma testemunha forte.

Como auditoria XAI adicional das relacoes espaciais, o CSV registra o par formado pelo objeto mais a esquerda e pelo objeto mais a direita de cada cena. As colunas `xai_leftmost_rightmost_conf`, `xai_reverse_conf` e `xai_asymmetry_conf` mostram, respectivamente, a confianca em `LeftOf(esquerda,direita)`, a confianca na relacao reversa e a satisfacao fuzzy da assimetria nesse par. Essa evidencia ajuda a explicar se o predicado horizontal aprendeu uma direcao coerente em exemplos concretos.

## 9. Discussao

O uso de formulas logicas permite avaliar nao apenas se os predicados acertam exemplos individuais, mas tambem se o conjunto de predicoes respeita propriedades globais do dominio. Por exemplo, um classificador puramente neural poderia aprender `LeftOf(x,y)` e `LeftOf(y,x)` como verdadeiros ao mesmo tempo em alguns casos. A regra de assimetria penaliza esse comportamento e induz uma estrutura mais coerente.

As consultas compostas testam a capacidade do modelo de combinar atributos e relacoes. A primeira consulta une tamanho, forma e relacoes verticais/horizontais. A segunda usa forma, cor e a relacao ternaria `InBetween`. A terceira testa uma implicacao condicional envolvendo proximidade e tamanho.

Como o treinamento usa uma unica cena, as metricas de classificacao avaliam uma generalizacao mais dificil: os predicados aprendidos precisam funcionar em configuracoes espaciais novas. Isso corresponde melhor ao objetivo do trabalho, que e raciocinio neuro-simbolico e nao apenas classificacao treinada com muitos exemplos.

## 10. Conclusao

O experimento mostra como LTN pode ser usado para combinar aprendizado neural com raciocinio logico diferenciavel. A representacao vetorial simplificada do CLEVR facilita a construcao de predicados para atributos e relacoes espaciais, enquanto a base de conhecimento permite impor propriedades como assimetria, transitividade, inversao de relacoes e regras compostas. Assim, o modelo nao apenas classifica objetos e relacoes, mas tambem pode ser auditado por perguntas logicas interpretaveis.

## Referencias

- Carraro, T., Serafini, L., & Aiolli, F. LTNtorch: PyTorch Implementation of Logic Tensor Networks.
- Johnson, J. et al. CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.
- Repositorio base usado como referencia: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

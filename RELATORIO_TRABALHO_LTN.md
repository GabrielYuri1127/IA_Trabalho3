# Trabalho Final de IA: Raciocínio Espacial Neuro-Simbólico com LTNtorch

Equipe: Gabriel Yuri, Guilherme Gurgel, Isabela Monteiro e Marcele Azevedo  
Disciplina: Inteligência Artificial  
Professor: Edjard Mota  
Tema: Logic Tensor Networks (LTN), CLEVR simplificado e raciocínio espacial  

## 1. Introdução: O que é NeSy e LTN?

A ideia da Inteligência Artificial Neuro-Simbólica (NeSy) é unir o melhor de dois mundos: o aprendizado das redes neurais com as regras claras da lógica simbólica. Em vez de treinar apenas um modelo do tipo entrada -> saída, uma abordagem NeSy permite que o treinamento seja guiado por conhecimento declarativo, como "se um objeto está à esquerda de outro, então o segundo não está à esquerda do primeiro".

No projeto, usamos Logic Tensor Networks (LTN) como uma ponte entre regras lógicas de domínio e Deep Learning. Em vez de deixar a rede neural aprender tudo do zero como uma caixa-preta, o treinamento é guiado por axiomas lógicos. Predicados como `LeftOf(x,y)` ou `IsSquare(x)` são implementados por redes neurais ou funções diferenciáveis que retornam valores de verdade fuzzy no intervalo `[0,1]`. Como essa lógica fuzzy é diferenciável, conseguimos treinar o modelo por retropropagação usando PyTorch.

No LTNtorch, esse processo segue cinco etapas principais: aterramento dos dados nas formulas, avaliacao dos valores de verdade, agregacao da satisfatibilidade, calculo da perda `1 - satAgg` e retropropagacao para ajustar os parametros dos predicados.

Essa etapa de aprendizado segue a mesma ideia do material complementar enviado pelo professor sobre calculo em redes neurais. Em uma rede comum, o neuronio calcula uma combinacao linear `z = W x + b`, aplica uma ativacao diferenciavel, por exemplo `sigma(z)`, mede uma perda e atualiza os pesos com descida de gradiente:

```text
W <- W - alpha * dLoss/dW
```

Na LTN, a diferenca e que a saida usada na perda nao e apenas uma classe prevista, mas a satisfatibilidade fuzzy das formulas logicas. Assim, a parte simbolica tambem participa do gradiente:

```text
Loss_total = Loss_fatos + lambda * (1 - satAgg)
```

No codigo, isso aparece no treinamento como `loss = facts + axiom_weight * (1 - sat)`, seguido de `loss.backward()` e `opt.step()`. Quando `LTNtorch` esta instalado, a funcao `ltn_training_sat` constroi um termo de `SatAgg` com `ltn.Variable`, `ltn.Predicate`, `ltn.Connective` e `ltn.Quantifier`, e esse termo tambem entra na perda de treinamento. O CSV registra `ltn_training_active=1` quando esse termo real do LTNtorch participou do treino. As formulas fuzzy locais permanecem como complemento e fallback. Como os predicados neurais, os conectivos fuzzy e o agregador `satAgg` sao diferenciaveis, o PyTorch consegue propagar o erro das formulas ate os pesos das redes dos predicados.

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

## 2. Dataset CLEVR Simplificado e Personalização

Trabalhar com imagens reais seria mais pesado para o processamento, então usamos uma versão simplificada inspirada no CLEVR. Em vez de processar imagens, cada objeto é representado por um vetor de 11 atributos. A personalização do trabalho foi trocar as formas originais de cilindro e cone por elipse e retângulo:

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

Para evitar sobreposicao visual entre objetos, a geracao das posicoes usa rejeicao amostral com duas condicoes: distancia minima de `0.08` entre centroides e teste geometrico de caixas visuais dos objetos desenhados. O CSV registra `min_pair_distance`, `overlap_ok`, `min_bbox_gap` e `bbox_overlap_ok` para auditar esse tratamento.

O predicado `CloseTo(x,y)` usa um criterio unico em todo o experimento: distancia euclidiana menor que `0.25`, excluindo o par `(x,x)`. O mesmo criterio e usado no treinamento supervisionado, na avaliacao e na consulta q3. O CSV registra `close_threshold` e `close_training_aligned` para auditar esse alinhamento.

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
CanStack(x,y) -> Above(x,y)
CanStack(x,y) -> not(IsRectangle(y) or IsTriangle(y))
CanStack(x,y) -> SameSize(x,y) or HorizAligned(x,y)
```

No treinamento supervisionado dos predicados, `CanStack(x,y)` considera a definicao completa usada no dominio: o objeto `x` deve estar acima do suporte `y`; o suporte `y` nao pode ser retangulo nem triangulo; e os objetos devem ter o mesmo tamanho ou centroide horizontal suficientemente alinhado.

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
  --overlap-margin 0.012 \
  --out resultados_clevr_ltn.csv --plot-dir figuras
```

| Execucao | Train seed | Test seed | satAgg | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2025 | 42 | 0.6081 | 0.8123 | 0.6759 | 0.6925 | 0.6841 |
| 2 | 2025 | 43 | 0.6139 | 0.8159 | 0.6614 | 0.7519 | 0.7037 |
| 3 | 2025 | 44 | 0.6142 | 0.7923 | 0.6180 | 0.7223 | 0.6661 |
| 4 | 2025 | 45 | 0.6201 | 0.8179 | 0.7073 | 0.6291 | 0.6659 |
| 5 | 2025 | 46 | 0.6100 | 0.7943 | 0.6242 | 0.7397 | 0.6770 |

Resumo:

| Metrica | Media | Desvio padrao |
|---|---:|---:|
| satAgg | 0.6133 | 0.0041 |
| Accuracy relacional | 0.8065 | 0.0110 |
| Precision relacional | 0.6573 | 0.0332 |
| Recall relacional | 0.7071 | 0.0438 |
| F1 relacional | 0.6794 | 0.0140 |
| Accuracy unaria | 0.9968 | 0.0030 |
| F1 unario | 0.9946 | 0.0051 |
| Accuracy geral | 0.8089 | 0.0109 |
| F1 geral | 0.6832 | 0.0137 |

Contagem de classes nas cenas de teste:

| Execucao | Circulos | Quadrados | Elipses | Retangulos | Triangulos |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 5 | 5 |
| 2 | 5 | 5 | 5 | 5 | 5 |
| 3 | 5 | 5 | 5 | 5 | 5 |
| 4 | 5 | 5 | 5 | 5 | 5 |
| 5 | 5 | 5 | 5 | 5 | 5 |

Tratamento de overlapping:

| Execucao | Test seed | Distancia minima entre centroides | Folga geometrica minima | `overlap_ok` | `bbox_overlap_ok` |
|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.0828 | 0.0146 | 1 | 1 |
| 2 | 43 | 0.0852 | 0.0139 | 1 | 1 |
| 3 | 44 | 0.0812 | 0.0140 | 1 | 1 |
| 4 | 45 | 0.0863 | 0.0141 | 1 | 1 |
| 5 | 46 | 0.0931 | 0.0124 | 1 | 1 |

Satisfatibilidade media das formulas e perguntas:

| Coluna no CSV | Formula/pergunta | Media | Desvio padrao |
|---|---|---:|---:|
| `shape_unique` | Forma unica | 0.9992 | 0.0003 |
| `shape_coverage` | Cobertura de formas | 0.8707 | 0.0535 |
| `left_irreflexive` | LeftOf irreflexivo | 0.9694 | 0.0043 |
| `left_asymmetric` | LeftOf assimetrico | 0.9078 | 0.0166 |
| `left_right_inverse` | Left/Right inversos | 0.8604 | 0.0058 |
| `left_transitive` | LeftOf transitivo | 0.9734 | 0.0050 |
| `below_above_inverse` | Below/Above inversos | 0.8673 | 0.0068 |
| `below_transitive` | Below transitivo | 0.9878 | 0.0068 |
| `between_rule` | Regra InBetween | 0.6253 | 0.0111 |
| `last_left` | Objeto mais a esquerda | 0.4162 | 0.0146 |
| `last_right` | Objeto mais a direita | 0.4237 | 0.0081 |
| `can_stack_above_rule` | Regra CanStack - objeto acima | 0.9299 | 0.0078 |
| `can_stack_rule` | Regra CanStack - suporte valido | 0.9994 | 0.0002 |
| `can_stack_stability_rule` | Regra CanStack - estabilidade | 0.7988 | 0.0144 |
| `query_small_below_ellipse_left_square` | Pergunta composta 1 | 0.0514 | 0.0112 |
| `query_green_rectangle_between` | Pergunta composta 2 | 0.1568 | 0.0185 |
| `query_triangles_close_same_size` | Pergunta composta 3 | 0.9417 | 0.0308 |
| `ltn_api_sat_check` | Auditoria complementar via LTNtorch | 0.9397 | 0.0073 |

Evidencia interpretavel das consultas compostas:

| Execucao | Test seed | q1 GT | q1 melhor testemunha | q1 objeto | q2 GT | q2 melhor testemunha | q2 objeto | q3 pares proximos | q3 violacoes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 1 | 0.9963 | 13 | 1 | 0.9981 | 10 | 2 | 0 |
| 2 | 43 | 1 | 0.9864 | 9 | 1 | 0.9992 | 22 | 6 | 6 |
| 3 | 44 | 1 | 0.9944 | 8 | 1 | 0.9959 | 15 | 2 | 0 |
| 4 | 45 | 1 | 0.9967 | 18 | 1 | 0.9991 | 10 | 0 | 0 |
| 5 | 46 | 1 | 0.9982 | 21 | 1 | 0.9993 | 18 | 2 | 2 |

Auditoria XAI do par horizontal extremo:

| Execucao | Test seed | Objeto mais a esquerda | Objeto mais a direita | `LeftOf(esq,dir)` | `LeftOf(dir,esq)` | Assimetria |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 18 | 24 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 43 | 4 | 20 | 1.0000 | 0.0000 | 1.0000 |
| 3 | 44 | 13 | 1 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 45 | 2 | 0 | 1.0000 | 0.0000 | 1.0000 |
| 5 | 46 | 17 | 1 | 1.0000 | 0.0000 | 1.0000 |

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

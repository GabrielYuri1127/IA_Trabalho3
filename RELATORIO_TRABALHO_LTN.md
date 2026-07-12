# Raciocinio Espacial Neuro-Simbolico com LTNtorch

Disciplina: Fundamentos de Inteligencia Artificial  
Tema: Logic Tensor Networks (LTN), CLEVR simplificado e raciocinio espacial  

## 1. Introducao: IA Neuro-Simbolica e LTN

A Inteligencia Artificial neuro-simbolica (NeSy) combina redes neurais, que aprendem padroes a partir de dados, com representacoes simbolicas, que permitem expressar conhecimento por regras logicas. Em vez de treinar apenas um modelo do tipo entrada -> saida, uma abordagem NeSy permite que o treinamento seja guiado por conhecimento declarativo, como "se um objeto esta a esquerda de outro, entao o segundo nao esta a esquerda do primeiro".

Logic Tensor Networks (LTN) sao uma familia de modelos neuro-simbolicos que mapeiam elementos da logica de primeira ordem para tensores diferenciaveis. Predicados como `LeftOf(x,y)` ou `IsSquare(x)` sao implementados por redes neurais ou funcoes diferenciaveis que retornam valores de verdade fuzzy no intervalo `[0,1]`. As formulas logicas sao avaliadas com operadores fuzzy, e a satisfatibilidade da base de conhecimento e usada como funcao objetivo durante o treinamento.

No LTNtorch, esse processo segue cinco etapas principais: aterramento dos dados nas formulas, avaliacao dos valores de verdade, agregacao da satisfatibilidade, calculo da perda `1 - satAgg` e retropropagacao para ajustar os parametros dos predicados.

## 2. Dataset CLEVR Simplificado

O trabalho usa uma versao simplificada do CLEVR. Em vez de processar imagens reais, cada objeto e representado por um vetor de 11 atributos:

| Posicoes | Significado |
|---|---|
| `[0,1]` | Coordenadas normalizadas `(x,y)` |
| `[2,3,4]` | Cor one-hot: vermelho, verde, azul |
| `[5,6,7,8,9]` | Forma one-hot: circulo, quadrado, cilindro, cone, triangulo |
| `[10]` | Tamanho: pequeno `0.0`, grande `1.0` |

Em cada execucao foram gerados 25 objetos aleatorios. O experimento foi repetido 5 vezes com sementes diferentes, conforme solicitado no enunciado.

## 3. Predicados Implementados

Predicados unarios:

- `IsCircle(x)`, `IsSquare(x)`, `IsCylinder(x)`, `IsCone(x)`, `IsTriangle(x)`
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
forall x: IsCircle(x) or IsSquare(x) or IsCylinder(x) or IsCone(x) or IsTriangle(x)
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
CanStack(x,y) -> not(IsCone(y) or IsTriangle(y))
```

No treinamento supervisionado dos predicados, `CanStack(x,y)` tambem considera a estabilidade pedida no enunciado: o suporte `y` nao pode ser cone nem triangulo e os objetos devem ter o mesmo tamanho ou centroide horizontal suficientemente alinhado.

## 5. Consultas Compostas

Consulta 1: existe algum objeto pequeno abaixo de um cilindro e a esquerda de um quadrado?

```text
exists x: IsSmall(x) and exists y: IsCylinder(y) and Below(x,y)
          and exists z: IsSquare(z) and LeftOf(x,z)
```

Consulta 2: existe um cone verde entre dois objetos quaisquer?

```text
exists x,y,z: IsCone(x) and IsGreen(x) and InBetween(x,y,z)
```

Consulta 3: se dois triangulos estao proximos, entao possuem o mesmo tamanho?

```text
forall x,y: (IsTriangle(x) and IsTriangle(y) and CloseTo(x,y)) -> SameSize(x,y)
```

## 6. Metricas

As metricas foram calculadas a partir da matriz de confusao:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 * Precision * Recall / (Precision + Recall)
```

Tambem foi reportado o valor de satisfatibilidade global `satAgg`, que indica o quanto a base de conhecimento esta sendo satisfeita pelo modelo.

## 7. Resultados

O experimento completo foi executado com 5 sementes aleatorias, 25 objetos por cena e 350 epocas por execucao. O arquivo completo gerado esta em `resultados_clevr_ltn.csv`.

Comando usado:

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --out resultados_clevr_ltn.csv --plot-dir figuras
```

| Execucao | Seed | satAgg | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.5914 | 0.9460 | 0.9217 | 0.8913 | 0.9062 |
| 2 | 43 | 0.5989 | 0.9387 | 0.8869 | 0.8948 | 0.8908 |
| 3 | 44 | 0.6165 | 0.9393 | 0.8892 | 0.9046 | 0.8968 |
| 4 | 45 | 0.6097 | 0.9277 | 0.8696 | 0.8744 | 0.8720 |
| 5 | 46 | 0.6170 | 0.9361 | 0.8516 | 0.9307 | 0.8894 |

Resumo:

| Metrica | Media | Desvio padrao |
|---|---:|---:|
| satAgg | 0.6067 | 0.0101 |
| Accuracy | 0.9376 | 0.0059 |
| Precision | 0.8838 | 0.0233 |
| Recall | 0.8991 | 0.0185 |
| F1 | 0.8910 | 0.0112 |

Satisfatibilidade media das formulas e perguntas:

| Coluna no CSV | Formula/pergunta | Media | Desvio padrao |
|---|---|---:|---:|
| `shape_unique` | Forma unica | 0.9998 | 0.0000 |
| `shape_coverage` | Cobertura de formas | 0.9992 | 0.0000 |
| `left_irreflexive` | LeftOf irreflexivo | 0.9726 | 0.0054 |
| `left_asymmetric` | LeftOf assimetrico | 0.9469 | 0.0051 |
| `left_right_inverse` | Left/Right inversos | 0.9084 | 0.0134 |
| `left_transitive` | LeftOf transitivo | 0.9878 | 0.0013 |
| `below_above_inverse` | Below/Above inversos | 0.9483 | 0.0084 |
| `below_transitive` | Below transitivo | 0.9946 | 0.0023 |
| `between_rule` | Regra InBetween | 0.7694 | 0.0075 |
| `last_left` | Objeto mais a esquerda | 0.4439 | 0.0045 |
| `last_right` | Objeto mais a direita | 0.4413 | 0.0123 |
| `can_stack_rule` | Regra CanStack | 0.9993 | 0.0004 |
| `query_small_below_cylinder_left_square` | Pergunta composta 1 | 0.0572 | 0.0234 |
| `query_green_cone_between` | Pergunta composta 2 | 0.1452 | 0.0825 |
| `query_triangles_close_same_size` | Pergunta composta 3 | 0.9426 | 0.0421 |
| `ltn_api_sat_check` | Auditoria complementar via LTNtorch | 0.6120 | 0.0147 |

Os valores baixos nas consultas existenciais compostas indicam que, nos cenarios aleatorios gerados, nem sempre existe uma configuracao que satisfaca simultaneamente todas as restricoes. Isso e esperado em dados aleatorios e faz parte da interpretacao logica da consulta.

Figuras geradas para visualizacao das cenas aleatorias:

| Seed | Figura |
|---:|---|
| 42 | [scene_seed_42.png](figuras/scene_seed_42.png) |
| 43 | [scene_seed_43.png](figuras/scene_seed_43.png) |
| 44 | [scene_seed_44.png](figuras/scene_seed_44.png) |
| 45 | [scene_seed_45.png](figuras/scene_seed_45.png) |
| 46 | [scene_seed_46.png](figuras/scene_seed_46.png) |

## 8. Explicacao do Raciocinio das Perguntas

Pergunta composta 1: o modelo procura um objeto `x` pequeno. Depois verifica se existe um objeto `y` que seja cilindro e esteja acima de `x`, isto e, `Below(x,y)`. Por fim, procura um objeto `z` que seja quadrado e esteja a direita de `x`, isto e, `LeftOf(x,z)`. A satisfatibilidade depende de as tres condicoes aparecerem juntas na mesma cena.

Pergunta composta 2: o modelo procura um objeto `x` que seja simultaneamente cone e verde. Em seguida, usa `InBetween(x,y,z)` para verificar se esse cone fica entre outros dois objetos no eixo horizontal. Como a cena e aleatoria, a consulta pode ter baixa satisfatibilidade quando nao ha cone verde em posicao intermediaria.

Pergunta composta 3: o modelo avalia todos os pares de objetos. Quando dois objetos sao triangulos e estao proximos segundo `CloseTo(x,y)`, a regra exige que `SameSize(x,y)` tambem seja verdadeiro. Esta e uma implicacao universal; portanto, ela testa consistencia global da cena, nao apenas a existencia de um exemplo.

## 9. Discussao

O uso de formulas logicas permite avaliar nao apenas se os predicados acertam exemplos individuais, mas tambem se o conjunto de predicoes respeita propriedades globais do dominio. Por exemplo, um classificador puramente neural poderia aprender `LeftOf(x,y)` e `LeftOf(y,x)` como verdadeiros ao mesmo tempo em alguns casos. A regra de assimetria penaliza esse comportamento e induz uma estrutura mais coerente.

As consultas compostas testam a capacidade do modelo de combinar atributos e relacoes. A primeira consulta une tamanho, forma e relacoes verticais/horizontais. A segunda usa forma, cor e a relacao ternaria `InBetween`. A terceira testa uma implicacao condicional envolvendo proximidade e tamanho.

Em geral, formulas ligadas diretamente ao vetor de atributos, como forma e tamanho, tiveram satisfatibilidade alta. Relacoes espaciais compostas tendem a ser mais dificeis, especialmente `InBetween` e regras com quantificadores existenciais ou universais aninhados, pois pequenos erros em predicados basicos podem se propagar para a formula composta.

## 10. Conclusao

O experimento mostra como LTN pode ser usado para combinar aprendizado neural com raciocinio logico diferenciavel. A representacao vetorial simplificada do CLEVR facilita a construcao de predicados para atributos e relacoes espaciais, enquanto a base de conhecimento permite impor propriedades como assimetria, transitividade, inversao de relacoes e regras compostas. Assim, o modelo nao apenas classifica objetos e relacoes, mas tambem pode ser auditado por perguntas logicas interpretaveis.

## Referencias

- Carraro, T., Serafini, L., & Aiolli, F. LTNtorch: PyTorch Implementation of Logic Tensor Networks.
- Johnson, J. et al. CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.
- Repositorio base usado como referencia: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

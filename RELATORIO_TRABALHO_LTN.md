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

7. Inverso abaixo-acima:

```text
forall x,y: Below(x,y) <-> Above(y,x)
```

8. Transitividade vertical:

```text
forall x,y,z: (Below(x,y) and Below(y,z)) -> Below(x,z)
```

9. Objeto entre dois outros:

```text
InBetween(x,y,z) <-> (LeftOf(y,x) and RightOf(z,x)) or (LeftOf(z,x) and RightOf(y,x))
```

10. Regra de empilhamento:

```text
CanStack(x,y) -> not(IsCone(y) or IsTriangle(y))
```

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

Preencha esta tabela com os valores gerados por:

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --out resultados_clevr_ltn.csv --plot-dir figuras
```

| Execucao | Seed | satAgg | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | preencher | preencher | preencher | preencher | preencher |
| 2 | 43 | preencher | preencher | preencher | preencher | preencher |
| 3 | 44 | preencher | preencher | preencher | preencher | preencher |
| 4 | 45 | preencher | preencher | preencher | preencher | preencher |
| 5 | 46 | preencher | preencher | preencher | preencher | preencher |

Resumo:

| Metrica | Media | Desvio padrao |
|---|---:|---:|
| satAgg | preencher | preencher |
| Accuracy | preencher | preencher |
| Precision | preencher | preencher |
| Recall | preencher | preencher |
| F1 | preencher | preencher |

O CSV tambem traz a satisfatibilidade de cada formula individual, como `shape_unique`, `left_asymmetric`, `left_transitive`, `below_transitive`, `between_rule` e as tres consultas compostas.

## 8. Discussao

O uso de formulas logicas permite avaliar nao apenas se os predicados acertam exemplos individuais, mas tambem se o conjunto de predicoes respeita propriedades globais do dominio. Por exemplo, um classificador puramente neural poderia aprender `LeftOf(x,y)` e `LeftOf(y,x)` como verdadeiros ao mesmo tempo em alguns casos. A regra de assimetria penaliza esse comportamento e induz uma estrutura mais coerente.

As consultas compostas testam a capacidade do modelo de combinar atributos e relacoes. A primeira consulta une tamanho, forma e relacoes verticais/horizontais. A segunda usa forma, cor e a relacao ternaria `InBetween`. A terceira testa uma implicacao condicional envolvendo proximidade e tamanho.

Em geral, espera-se que formulas ligadas diretamente ao vetor de atributos, como forma e tamanho, tenham satisfatibilidade alta. Relacoes espaciais compostas tendem a ser mais dificeis, especialmente `InBetween` e regras com quantificadores existenciais ou universais aninhados, pois pequenos erros em predicados basicos podem se propagar para a formula composta.

## 9. Conclusao

O experimento mostra como LTN pode ser usado para combinar aprendizado neural com raciocinio logico diferenciavel. A representacao vetorial simplificada do CLEVR facilita a construcao de predicados para atributos e relacoes espaciais, enquanto a base de conhecimento permite impor propriedades como assimetria, transitividade, inversao de relacoes e regras compostas. Assim, o modelo nao apenas classifica objetos e relacoes, mas tambem pode ser auditado por perguntas logicas interpretaveis.

## Referencias

- Carraro, T., Serafini, L., & Aiolli, F. LTNtorch: PyTorch Implementation of Logic Tensor Networks.
- Johnson, J. et al. CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.
- Repositorio base usado como referencia: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

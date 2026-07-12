# Trabalho Final de IA: Raciocínio Espacial Neuro-Simbólico com LTNtorch

Este repositório contém a implementação do trabalho final da disciplina **Inteligência Artificial**, cujo objetivo é construir e avaliar um modelo neuro-simbólico para **raciocínio espacial** em cenas 2D inspiradas no dataset CLEVR.

**Equipe:** Gabriel Yuri, Guilherme Gurgel, Isabela Monteiro e Marcele Azevedo  
**Disciplina:** Inteligência Artificial  
**Professor:** Edjard Mota  
**Tema:** Logic Tensor Networks (LTN), CLEVR simplificado e raciocínio espacial

---

## 1. Visão Geral

A proposta utiliza **Logic Tensor Networks (LTN)** para combinar redes neurais com regras lógicas diferenciáveis. O foco não é apenas classificar objetos, mas verificar se regras de **reasoning** podem ser aprendidas a partir de poucos dados e generalizadas para novas cenas.

No experimento, o modelo aprende predicados neurais para atributos e relações espaciais, enquanto axiomas lógicos orientam o treinamento por meio da satisfatibilidade fuzzy (`satAgg`). Assim, propriedades como assimetria, transitividade e inversão de relações passam a influenciar a função de perda.

---

## 2. Dataset CLEVR Simplificado

Como o processamento de imagens reais seria mais pesado, foi usada uma versão vetorial simplificada do CLEVR. Cada objeto é representado por um vetor de 11 atributos:

| Posições | Significado |
|---|---|
| `[0, 1]` | Coordenadas normalizadas `(x, y)` |
| `[2, 3, 4]` | Cor one-hot: vermelho, verde e azul |
| `[5, 6, 7, 8, 9]` | Forma one-hot: círculo, quadrado, elipse, retângulo e triângulo |
| `[10]` | Tamanho: pequeno `0.0` ou grande `1.0` |

### 2.1 Adaptação das formas

Por se tratar de um cenário 2D, as formas originalmente associadas a objetos 3D foram adaptadas:

```text
cilindro -> elipse
cone     -> retângulo
```

As cinco classes usadas no trabalho são:

| Classe | Quantidade por cena |
|---|---:|
| Círculo | 5 |
| Quadrado | 5 |
| Elipse | 5 |
| Retângulo | 5 |
| Triângulo | 5 |

Total por cena: **25 objetos**.

### 2.2 Geração das cenas

O professor esclareceu que não são cinco objetos no total, mas sim cinco classes, com cinco objetos de cada classe. Por isso, o protocolo experimental usa:

- **1 cena de treino**, com 25 objetos balanceados;
- **5 cenas de teste**, aleatórias e distintas;
- **5 objetos de cada classe** em todas as cenas;
- tratamento de overlapping por distância entre centroides e por caixas geométricas visuais.

---

## 3. Predicados e Regras Implementadas

O modelo implementa predicados unários, binários e ternários para representar atributos e relações espaciais.

### 3.1 Predicados de atributos

- `IsCircle(x)`
- `IsSquare(x)`
- `IsEllipse(x)`
- `IsRectangle(x)`
- `IsTriangle(x)`
- `IsSmall(x)` / `IsBig(x)`
- `IsRed(x)` / `IsGreen(x)` / `IsBlue(x)`

### 3.2 Predicados espaciais

- `LeftOf(x, y)`
- `RightOf(x, y)`
- `Below(x, y)`
- `Above(x, y)`
- `CloseTo(x, y)`
- `InBetween(x, y, z)`
- `CanStack(x, y)`

### 3.3 Axiomas principais

As regras lógicas avaliadas no experimento incluem:

- unicidade e cobertura de formas;
- irreflexividade, assimetria e transitividade de `LeftOf`;
- relação inversa entre `LeftOf` e `RightOf`;
- relação inversa entre `Below` e `Above`;
- transitividade vertical;
- regra ternária para `InBetween`;
- definição completa de `CanStack`;
- consultas compostas para avaliar raciocínio relacional.

A definição de `CanStack(x, y)` considera três condições: o objeto `x` deve estar acima de `y`, o suporte `y` não pode ser retângulo nem triângulo, e os objetos devem ter mesmo tamanho ou alinhamento horizontal suficiente.

---

## 4. Uso do LTNtorch no Treinamento

O treinamento combina erro supervisionado dos predicados com perda lógica baseada em satisfatibilidade:

```text
Loss_total = Loss_fatos + lambda * (1 - satAgg)
```

Quando `LTNtorch` está instalado, a função `ltn_training_sat` constrói fórmulas diferenciáveis usando objetos reais da biblioteca:

- `ltn.Variable`
- `ltn.Predicate`
- `ltn.Connective`
- `ltn.Quantifier`
- `SatAgg`

Essas fórmulas participam diretamente da perda de treinamento. As fórmulas fuzzy locais permanecem como complemento e fallback para manter o experimento executável em ambientes diferentes.

---

## 5. Consultas Compostas

Além dos axiomas estruturais, o trabalho avalia perguntas de raciocínio espacial:

| Consulta | Ideia avaliada |
|---|---|
| q1 | Existe objeto pequeno abaixo de uma elipse e à esquerda de um quadrado? |
| q2 | Existe retângulo verde entre dois objetos? |
| q3 | Se dois triângulos estão próximos, então possuem o mesmo tamanho? |

Essas consultas exigem combinação de atributos, relações espaciais e implicações lógicas.

---

## 6. Resultados

O experimento completo foi executado com uma cena de treino (`train_seed=2025`) e cinco cenas de teste (`seed=42` a `seed=46`).

| Métrica | Média | Desvio padrão |
|---|---:|---:|
| `satAgg` | 0.6133 | 0.0041 |
| Accuracy relacional | 0.8065 | 0.0110 |
| Precision relacional | 0.6573 | 0.0332 |
| Recall relacional | 0.7071 | 0.0438 |
| F1 relacional | 0.6794 | 0.0140 |
| Accuracy unária | 0.9968 | 0.0030 |
| F1 unário | 0.9946 | 0.0051 |
| Accuracy geral | 0.8089 | 0.0109 |
| F1 geral | 0.6832 | 0.0137 |

O CSV final também registra:

- `min_bbox_gap` e `bbox_overlap_ok`, para auditar o tratamento geométrico de overlapping;
- `can_stack_above_rule`, `can_stack_rule` e `can_stack_stability_rule`, para auditar `CanStack`;
- `q1_*`, `q2_*` e `q3_*`, para evidências interpretáveis das consultas;
- `xai_*`, para análise do par mais à esquerda e mais à direita de cada cena.

---

## 7. Estrutura do Repositório

| Arquivo/Pasta | Descrição |
|---|---|
| `clevr_ltn_experimentos.py` | Script principal do experimento |
| `trabalho_ltn_clevr.ipynb` | Notebook executável |
| `RELATORIO_TRABALHO_LTN.md` | Relatório completo em Markdown |
| `CHECKLIST_PROFESSOR.md` | Checklist de atendimento ao enunciado |
| `resultados_clevr_ltn.csv` | Resultados finais do experimento |
| `figuras/` | Cenas geradas para treino e teste |
| `.github/workflows/test.yml` | Teste rápido no GitHub Actions |
| `.github/workflows/full-experiment.yml` | Execução completa no GitHub Actions |

---

## 8. Como Executar

### 8.1 Instalar dependências

```bash
pip install torch numpy matplotlib LTNtorch
```

Se `LTNtorch` não instalar com esse nome no ambiente usado, tente:

```bash
pip install ltn
```

### 8.2 Rodar pelo notebook

Abra e execute as células do arquivo:

[trabalho_ltn_clevr.ipynb](trabalho_ltn_clevr.ipynb)

### 8.3 Rodar pelo terminal

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --train-seed 2025 --seed 42 --min-distance 0.08 --overlap-margin 0.012 --out resultados_clevr_ltn.csv --plot-dir figuras
```

Esse comando gera o CSV de resultados e as figuras das cenas.

---

## 9. Figuras

- [Cena de treino - seed 2025](figuras/scene_seed_2025.png)
- [Cena de teste - seed 42](figuras/scene_seed_42.png)
- [Cena de teste - seed 43](figuras/scene_seed_43.png)
- [Cena de teste - seed 44](figuras/scene_seed_44.png)
- [Cena de teste - seed 45](figuras/scene_seed_45.png)
- [Cena de teste - seed 46](figuras/scene_seed_46.png)

---

## 10. Materiais de Entrega

- [Relatório completo](RELATORIO_TRABALHO_LTN.md)
- [Checklist do enunciado](CHECKLIST_PROFESSOR.md)
- [CSV de resultados](resultados_clevr_ltn.csv)
- [Notebook executável](trabalho_ltn_clevr.ipynb)

Testes no GitHub Actions:

- [Teste rápido do experimento](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/test.yml)
- [Experimento completo LTN](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/full-experiment.yml)

---

## 11. Referências

- Carraro, T., Serafini, L., & Aiolli, F. **LTNtorch: PyTorch Implementation of Logic Tensor Networks**.
- Johnson, J. et al. **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning**.
- Repositório base: <https://github.com/edjard/Clevr_LTNtorch>
- Documentação LTNtorch: <https://tommasocarraro.github.io/LTNtorch/>

# Trabalho Final de Inteligência Artificial

## Raciocínio Espacial Neuro-Simbólico com LTNtorch

**Equipe:** Gabriel Yuri, Guilherme Gurgel, Isabela Monteiro e Marcele Azevedo  
**Disciplina:** Inteligência Artificial  
**Professor:** Edjard Mota  

Este repositório reúne o código, o relatório, o notebook, os resultados e as figuras do trabalho final da disciplina. O tema desenvolvido foi o uso de **Logic Tensor Networks (LTN)** para representar e avaliar regras de raciocínio espacial em uma versão simplificada do dataset CLEVR.

---

## 1. Objetivo do Trabalho

O objetivo do trabalho é construir um experimento neuro-simbólico capaz de combinar:

- redes neurais para aprender predicados sobre objetos;
- regras lógicas para representar conhecimento espacial;
- lógica fuzzy diferenciável para permitir treinamento por retropropagação;
- métricas de avaliação para verificar se as regras aprendidas generalizam para novas cenas.

Diferente de uma tarefa simples de classificação, o foco do trabalho é avaliar **raciocínio**. Por isso, o modelo é treinado com uma única cena e depois testado em cenas aleatórias diferentes.

---

## 2. IA Neuro-Simbólica e LTN

A Inteligência Artificial Neuro-Simbólica (NeSy) combina aprendizado neural com representações simbólicas. Neste trabalho, essa ideia é aplicada com **Logic Tensor Networks (LTN)**.

Na LTN, predicados como `LeftOf(x, y)`, `Below(x, y)` ou `IsTriangle(x)` retornam valores fuzzy no intervalo `[0, 1]`. As fórmulas lógicas também recebem valores de satisfatibilidade, agregados por `satAgg`.

Durante o treinamento, a perda combina fatos supervisionados e satisfatibilidade lógica:

```text
Loss_total = Loss_fatos + lambda * (1 - satAgg)
```

O treinamento exige `LTNtorch` e usa objetos da própria biblioteca (`ltn.Variable`, `ltn.Predicate`, `ltn.Connective`, `ltn.Quantifier` e `SatAgg`) como parte da perda de treinamento.

O CSV registra `ltn_training_active=1` quando esse termo real do LTNtorch participou do treinamento.

---

## 3. Dataset CLEVR Simplificado

O trabalho usa uma versão vetorial simplificada do CLEVR. Em vez de usar imagens reais, cada objeto é representado por um vetor com 11 atributos:

| Posições | Descrição |
|---|---|
| `[0, 1]` | Coordenadas normalizadas `(x, y)` |
| `[2, 3, 4]` | Cor one-hot: vermelho, verde e azul |
| `[5, 6, 7, 8, 9]` | Forma one-hot: círculo, quadrado, elipse, retângulo e triângulo |
| `[10]` | Tamanho: pequeno `0.0` ou grande `1.0` |

### 3.1 Alteração das formas

Como o experimento foi desenvolvido em um espaço 2D, as formas foram adaptadas:

```text
cilindro -> elipse
cone     -> retângulo
```

As classes usadas foram:

| Classe | Quantidade por cena |
|---|---:|
| Círculo | 5 |
| Quadrado | 5 |
| Elipse | 5 |
| Retângulo | 5 |
| Triângulo | 5 |

Cada cena possui, portanto, **25 objetos**.

### 3.2 Protocolo usado

Conforme a orientação do professor:

- não são 5 objetos no total;
- são 5 classes de objetos;
- cada classe aparece 5 vezes;
- o treino usa uma única cena;
- o teste usa 5 cenas aleatórias distintas;
- o objetivo é avaliar generalização de regras de raciocínio.

As cenas também usam tratamento de sobreposição visual por distância entre centroides e por caixas geométricas dos objetos.

O predicado `CloseTo(x, y)` usa o mesmo critério em todo o experimento: distância euclidiana menor que `0.25`, ignorando pares do tipo `(x, x)`. Esse critério é usado no treinamento, na avaliação e na consulta q3.

---

## 4. Predicados Implementados

### 4.1 Predicados de atributos

- `IsCircle(x)`
- `IsSquare(x)`
- `IsEllipse(x)`
- `IsRectangle(x)`
- `IsTriangle(x)`
- `IsSmall(x)` e `IsBig(x)`
- `IsRed(x)`, `IsGreen(x)` e `IsBlue(x)`

### 4.2 Predicados espaciais

- `LeftOf(x, y)`
- `RightOf(x, y)`
- `Below(x, y)`
- `Above(x, y)`
- `CloseTo(x, y)`
- `InBetween(x, y, z)`
- `CanStack(x, y)`

---

## 5. Regras Lógicas Avaliadas

O experimento avalia regras como:

- unicidade de forma;
- cobertura das formas;
- irreflexividade de `LeftOf`;
- assimetria de `LeftOf`;
- relação inversa entre `LeftOf` e `RightOf`;
- transitividade horizontal;
- relação inversa entre `Below` e `Above`;
- transitividade vertical;
- regra de objeto entre dois outros objetos (`InBetween`);
- definição completa de `CanStack`.

No trabalho, `CanStack(x, y)` considera que:

- `x` deve estar acima de `y`;
- `y` não pode ser retângulo nem triângulo;
- os objetos devem ter o mesmo tamanho ou alinhamento horizontal suficiente.

---

## 6. Consultas Compostas

Também foram avaliadas três consultas de raciocínio:

| Consulta | Pergunta |
|---|---|
| q1 | Existe objeto pequeno abaixo de uma elipse e à esquerda de um quadrado? |
| q2 | Existe retângulo verde entre dois objetos? |
| q3 | Se dois triângulos estão próximos, então possuem o mesmo tamanho? |

Essas consultas combinam atributos, relações espaciais e implicações lógicas.

---

## 7. Resultados Obtidos

O experimento completo foi executado com:

- cena de treino: `train_seed=2025`;
- cenas de teste: `seed=42`, `43`, `44`, `45` e `46`;
- 25 objetos por cena;
- 5 objetos de cada classe.

Resumo das métricas:

| Métrica | Média | Desvio padrão |
|---|---:|---:|
| `satAgg` | 0.6116 | 0.0038 |
| Accuracy relacional | 0.8221 | 0.0077 |
| Precision relacional | 0.7000 | 0.0306 |
| Recall relacional | 0.6842 | 0.0458 |
| F1 relacional | 0.6901 | 0.0166 |
| Accuracy unária | 0.9968 | 0.0030 |
| F1 unário | 0.9946 | 0.0051 |
| Accuracy geral | 0.8243 | 0.0076 |
| F1 geral | 0.6941 | 0.0162 |

O arquivo `resultados_clevr_ltn.csv` contém os resultados completos, incluindo:

- satisfatibilidade de cada fórmula;
- uso efetivo do LTNtorch no treinamento (`ltn_training_active` e `ltn_training_sat_final`);
- métricas por cena de teste;
- auditoria de overlapping (`min_bbox_gap` e `bbox_overlap_ok`);
- critério alinhado de proximidade (`close_threshold` e `close_training_aligned`);
- evidências interpretáveis das consultas;
- auditoria XAI do par mais à esquerda e mais à direita.

---

## 8. Arquivos do Repositório

| Arquivo/Pasta | Conteúdo |
|---|---|
| `clevr_ltn_experimentos.py` | Implementação principal do experimento |
| `trabalho_ltn_clevr.ipynb` | Notebook para execução e visualização |
| `RELATORIO_TRABALHO_LTN.md` | Relatório completo do trabalho |
| `CHECKLIST_PROFESSOR.md` | Checklist dos requisitos do enunciado |
| `resultados_clevr_ltn.csv` | Resultados finais do experimento |
| `requirements.txt` | Dependências Python do projeto |
| `figuras/` | Cenas de treino e teste geradas |
| `.github/workflows/test.yml` | Teste rápido no GitHub Actions |
| `.github/workflows/full-experiment.yml` | Execução completa no GitHub Actions |

---

## 9. Como Reproduzir

### 9.1 Instalar dependências

```bash
pip install torch numpy matplotlib LTNtorch
```

Caso `LTNtorch` não instale com esse nome no ambiente local, a alternativa abaixo pode ser testada, mas a execução usada na entrega instala `LTNtorch` explicitamente:

```bash
pip install ltn
```

### 9.2 Executar pelo notebook

Abra o notebook:

[trabalho_ltn_clevr.ipynb](trabalho_ltn_clevr.ipynb)

Execute as células em ordem.

### 9.3 Executar pelo terminal

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --train-seed 2025 --seed 42 --min-distance 0.08 --overlap-margin 0.012 --out resultados_clevr_ltn.csv --plot-dir figuras
```

Esse comando gera o CSV de resultados e as imagens das cenas.

---

## 10. Figuras Geradas

- [Cena de treino - seed 2025](figuras/scene_seed_2025.png)
- [Cena de teste - seed 42](figuras/scene_seed_42.png)
- [Cena de teste - seed 43](figuras/scene_seed_43.png)
- [Cena de teste - seed 44](figuras/scene_seed_44.png)
- [Cena de teste - seed 45](figuras/scene_seed_45.png)
- [Cena de teste - seed 46](figuras/scene_seed_46.png)

---

## 11. Referências

- Carraro, T., Serafini, L., & Aiolli, F. **LTNtorch: PyTorch Implementation of Logic Tensor Networks**.
- Johnson, J. et al. **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning**.
- Repositório base: <https://github.com/edjard/Clevr_LTNtorch>
- Documentação LTNtorch: <https://tommasocarraro.github.io/LTNtorch/>

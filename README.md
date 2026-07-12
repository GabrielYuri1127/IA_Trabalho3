# Trabalho 3 - Raciocinio Espacial Neuro-Simbolico com LTN

Este repositorio contem a implementacao e o relatorio do trabalho final de Fundamentos de Inteligencia Artificial sobre Logic Tensor Networks (LTN), raciocinio espacial e um dataset CLEVR simplificado.

## Ajuste conforme orientacao do professor

O dataset nao usa 5 objetos no total. Ele usa 5 classes de objetos, com 5 objetos de cada classe, totalizando 25 objetos por cena:

- 5 circulos
- 5 quadrados
- 5 elipses
- 5 retangulos
- 5 triangulos

O experimento treina os predicados e regras de reasoning em uma cena balanceada e depois testa a generalizacao em 5 cenas/datasets aleatorios distintos, tambem balanceados.

As consultas compostas cobrem os niveis de raciocinio pedidos no material da atividade: q1 e q2 sao de raciocinio espacial/relacional multi-hop, e q3 e uma implicacao material do nivel condicional.

## Arquivos principais

- `clevr_ltn_experimentos.py`: script do experimento CLEVR simplificado com treino unico e 5 testes aleatorios.
- `trabalho_ltn_clevr.ipynb`: notebook executavel para rodar o teste, o experimento completo, abrir metricas e visualizar figuras.
- `RELATORIO_TRABALHO_LTN.md`: relatorio com explicacao teorica, formulas e resultados finais.
- `CHECKLIST_PROFESSOR.md`: checklist item a item comparando a entrega com o enunciado.
- `resultados_clevr_ltn.csv`: metricas geradas pelo experimento completo.
- `figuras/`: imagens dos cenarios gerados.
- `.github/workflows/test.yml`: teste automatico rapido.
- `.github/workflows/full-experiment.yml`: experimento completo automatico.

## Como instalar

Use um ambiente Python novo ou Google Colab.

```bash
pip install torch numpy matplotlib LTNtorch
```

Se `LTNtorch` nao instalar com esse nome no ambiente usado, tente:

```bash
pip install ltn
```

## Como rodar pelo notebook

Abra o arquivo abaixo e execute as celulas em ordem:

- [trabalho_ltn_clevr.ipynb](trabalho_ltn_clevr.ipynb)

O notebook contem uma execucao curta de teste, a execucao completa, a leitura do CSV e a visualizacao das figuras.

## Como rodar o experimento completo pelo terminal

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --train-seed 2025 --seed 42 --out resultados_clevr_ltn.csv --plot-dir figuras
```

Isso gera:

- `resultados_clevr_ltn.csv`: metricas por cena de teste.
- `figuras/`: imagem da cena de treino e imagens das 5 cenas de teste.
- colunas de interpretabilidade das consultas: resposta ground-truth, melhor testemunha e confianca para q1/q2.

## Resultados e prints

Resultados finais:

- [Relatorio preenchido](RELATORIO_TRABALHO_LTN.md)
- [Checklist do enunciado](CHECKLIST_PROFESSOR.md)
- [CSV de resultados](resultados_clevr_ltn.csv)
- [Notebook executavel](trabalho_ltn_clevr.ipynb)

Figuras das cenas:

- [Cena de treino seed 2025](figuras/scene_seed_2025.png)
- [Cena teste seed 42](figuras/scene_seed_42.png)
- [Cena teste seed 43](figuras/scene_seed_43.png)
- [Cena teste seed 44](figuras/scene_seed_44.png)
- [Cena teste seed 45](figuras/scene_seed_45.png)
- [Cena teste seed 46](figuras/scene_seed_46.png)

Testes no GitHub Actions:

- [Teste rapido do experimento](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/test.yml)
- [Experimento completo LTN](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/full-experiment.yml)

Para documentar a execucao, recomenda-se tirar prints de:

1. Pagina principal do repositorio mostrando os arquivos.
2. `RELATORIO_TRABALHO_LTN.md`, principalmente a secao de resultados.
3. `CHECKLIST_PROFESSOR.md`, mostrando que os itens do enunciado foram atendidos.
4. `resultados_clevr_ltn.csv` aberto no GitHub.
5. Uma ou mais imagens da pasta `figuras/`.
6. GitHub Actions com o teste rapido e o experimento completo marcados como sucesso.

## Referencias

- Repositorio base: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

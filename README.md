# Trabalho 3 - Raciocinio Espacial Neuro-Simbolico com LTN

Este repositorio contem a implementacao e o relatorio do trabalho final de Fundamentos de Inteligencia Artificial sobre Logic Tensor Networks (LTN), raciocinio espacial e um dataset CLEVR simplificado.

## Arquivos principais

- `clevr_ltn_experimentos.py`: script dos 5 experimentos com CLEVR simplificado.
- `RELATORIO_TRABALHO_LTN.md`: relatorio em Markdown com explicacao teorica, formulas e tabela de resultados.
- `resultados_clevr_ltn.csv`: arquivo gerado ao rodar os experimentos.
- `figuras/`: imagens dos cenarios aleatorios, geradas opcionalmente.

## 1. Instalar dependencias

Use um ambiente Python novo ou Google Colab.

```bash
pip install torch numpy matplotlib LTNtorch
```

Se `LTNtorch` nao instalar com esse nome no seu ambiente, tente:

```bash
pip install ltn
```

## 2. Rodar experimentos

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --out resultados_clevr_ltn.csv --plot-dir figuras
```

Isso gera:

- `resultados_clevr_ltn.csv`: metricas por execucao.
- `figuras/`: imagens dos 5 cenarios aleatorios, se `matplotlib` estiver instalado.

## 3. Preencher relatorio

Abra `resultados_clevr_ltn.csv` e copie os valores para a tabela da secao "Resultados" em `RELATORIO_TRABALHO_LTN.md`.

## Referencias

- Repositorio base: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

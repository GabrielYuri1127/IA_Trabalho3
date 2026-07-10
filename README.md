# Trabalho 3 - Raciocinio Espacial Neuro-Simbolico com LTN

Este repositorio contem a implementacao e o relatorio do trabalho final de Fundamentos de Inteligencia Artificial sobre Logic Tensor Networks (LTN), raciocinio espacial e um dataset CLEVR simplificado.

## Arquivos principais

- `clevr_ltn_experimentos.py`: script dos 5 experimentos com CLEVR simplificado.
- `RELATORIO_TRABALHO_LTN.md`: relatorio com explicacao teorica, formulas e resultados finais.
- `resultados_clevr_ltn.csv`: metricas geradas pelo experimento completo.
- `figuras/`: imagens dos 5 cenarios aleatorios gerados.
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

## Como rodar o experimento completo

```bash
python clevr_ltn_experimentos.py --runs 5 --epochs 350 --out resultados_clevr_ltn.csv --plot-dir figuras
```

Isso gera:

- `resultados_clevr_ltn.csv`: metricas por execucao.
- `figuras/`: imagens dos 5 cenarios aleatorios.

## Resultados e prints

Resultados finais:

- [Relatorio preenchido](RELATORIO_TRABALHO_LTN.md)
- [CSV de resultados](resultados_clevr_ltn.csv)

Figuras das cenas:

- [Cena seed 42](figuras/scene_seed_42.png)
- [Cena seed 43](figuras/scene_seed_43.png)
- [Cena seed 44](figuras/scene_seed_44.png)
- [Cena seed 45](figuras/scene_seed_45.png)
- [Cena seed 46](figuras/scene_seed_46.png)

Testes no GitHub Actions:

- [Teste rapido do experimento](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/test.yml)
- [Experimento completo LTN](https://github.com/GabrielYuri1127/IA_Trabalho3/actions/workflows/full-experiment.yml)

Para documentar a execucao, recomenda-se tirar prints de:

1. Pagina principal do repositorio mostrando os arquivos.
2. `RELATORIO_TRABALHO_LTN.md`, principalmente a secao de resultados.
3. `resultados_clevr_ltn.csv` aberto no GitHub.
4. Uma ou mais imagens da pasta `figuras/`.
5. GitHub Actions com o teste rapido e o experimento completo marcados como sucesso.

## Referencias

- Repositorio base: https://github.com/edjard/Clevr_LTNtorch
- Documentacao LTNtorch: https://tommasocarraro.github.io/LTNtorch/

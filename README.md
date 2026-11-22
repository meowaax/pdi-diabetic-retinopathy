# Segmentação multiclasse de lesões da retinopatia diabética (IDRiD)

UNet++ com encoder VGG16 comparando RGB, CLAHE e canal verde. Planejamento em
`planejamento_projeto_pdi_retinopatia_diabetica.md`.

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # versões testadas em requirements.lock.txt
python scripts/check_env.py          # confere bibliotecas, dispositivo e forward/backward
```

Dispositivo detectado automaticamente: CUDA, MPS (Apple Silicon) ou CPU (`src/seed.py`).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `data/` | IDRiD (não versionado): `data/A. Segmentation/` |
| `src/` | código reutilizável (dataset, pré-processamento, modelo, loss, métricas) |
| `scripts/` | scripts de linha de comando |
| `notebooks/` | exploração e análises |
| `configs/` | configurações YAML dos experimentos |
| `results/` | saídas (não versionadas, exceto `.gitkeep`) |

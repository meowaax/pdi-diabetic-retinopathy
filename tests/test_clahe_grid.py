from pathlib import Path

import yaml

from scripts.make_clahe_grid import CLIP_LIMITS, TILE_SIZES, build_grid


def test_grid_tem_8_arquivos_com_os_parametros_esperados(tmp_path):
    paths = build_grid(out_dir=tmp_path)
    assert len(paths) == len(CLIP_LIMITS) * len(TILE_SIZES) == 8

    vistos = set()
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        assert cfg["preprocess_mode"] == "clahe"
        clip = cfg["preprocess_params"]["clip_limit"]
        tile = tuple(cfg["preprocess_params"]["tile_grid_size"])
        assert clip in CLIP_LIMITS and tile[0] in TILE_SIZES and tile[0] == tile[1]
        vistos.add((clip, tile[0]))
    assert len(vistos) == 8  # sem duplicatas


def test_grid_herda_hiperparametros_do_yaml_base(tmp_path):
    base = yaml.safe_load(Path("configs/train_default.yaml").read_text())
    paths = build_grid(out_dir=tmp_path)
    cfg = yaml.safe_load(paths[0].read_text())
    assert cfg["optim"]["lr"] == base["optim"]["lr"]
    assert cfg["optim"]["batch_size"] == base["optim"]["batch_size"]
    assert cfg["train"]["patience"] == base["train"]["patience"]


def test_arquivos_versionados_batem_com_a_funcao_build_grid():
    """configs/clahe_grid/ já commitado deve estar em sincronia com o gerador."""
    versionados = sorted(p.name for p in Path("configs/clahe_grid").glob("*.yaml"))
    gerados = sorted(p.name for p in build_grid(out_dir=Path("/tmp/clahe_grid_check")))
    assert versionados == gerados

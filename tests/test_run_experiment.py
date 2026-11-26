"""Teste de integração do CLI scripts/run_experiment.py (smoke test, poucas imagens/épocas)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/cache/IDRiD_01_meta.npz").exists(), reason="rode scripts/build_cache.py")


def test_run_experiment_subset_produz_json_de_resultado(tmp_path, monkeypatch):
    cfg_path = tmp_path / "smoke.yaml"
    cfg = yaml.safe_load((ROOT / "configs/train_default.yaml").read_text())
    cfg["model"]["encoder_weights"] = None
    cfg["optim"]["batch_size"] = 1
    cfg["train"]["patches_per_image"] = 2
    cfg["eval"]["stride"] = 2048
    cfg["paths"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["paths"]["log_dir"] = str(tmp_path / "logs")
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.chdir(ROOT)
    proc = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", str(cfg_path), "--subset",
         "--run-name", "smoke_pytest", "--max-epochs", "2"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    out = ROOT / "results/runs/smoke_pytest.json"
    assert out.exists()
    result = json.loads(out.read_text())
    assert "best_val_macro_dice" in result and "per_class_dice" in result
    assert result["selection"] == {"mode": "subset"}
    out.unlink()  # smoke test não deve deixar artefato versionável

import json
from pathlib import Path

from scripts.collect_results import base_name, load_runs, summarize_folds, summarize_subset


def _write_run(dir_, name, mode, fold=None, macro_dice=0.5, dice=None):
    dice = dice or {"MA": 0.1, "HE": 0.2, "EX": 0.3, "SE": 0.4}
    r = {"best_epoch": 3, "best_val_macro_dice": macro_dice, "per_class_dice": dice,
         "selection": {"mode": mode, **({"fold": fold} if fold is not None else {})}}
    (dir_ / f"{name}.json").write_text(json.dumps(r))


def test_base_name_remove_sufixo_de_dobra():
    assert base_name("e1_rgb_fold3") == "e1_rgb"
    assert base_name("clahe_clip2_tile8") == "clahe_clip2_tile8"  # sem sufixo, intacto


def test_load_runs_le_json_e_preenche_dice_por_classe(tmp_path):
    _write_run(tmp_path, "run_a", "subset", macro_dice=0.6)
    df = load_runs(tmp_path)
    assert len(df) == 1 and df.iloc[0].macro_dice == 0.6 and df.iloc[0].dice_MA == 0.1


def test_summarize_subset_ordena_por_macro_dice_desc(tmp_path):
    _write_run(tmp_path, "a", "subset", macro_dice=0.3)
    _write_run(tmp_path, "b", "subset", macro_dice=0.7)
    df = load_runs(tmp_path)
    out = summarize_subset(df)
    assert list(out.run_name) == ["b", "a"]


def test_summarize_folds_agrega_media_e_desvio(tmp_path):
    for k, md in enumerate([0.5, 0.6, 0.7]):
        _write_run(tmp_path, f"e1_rgb_fold{k}", "fold", fold=k, macro_dice=md)
    df = load_runs(tmp_path)
    out = summarize_folds(df)
    assert out.experiment.tolist() == ["e1_rgb"]
    row = out.iloc[0]
    assert row.macro_dice_count == 3
    assert abs(row.macro_dice_mean - 0.6) < 1e-9


def test_load_runs_sem_arquivos_devolve_dataframe_vazio_com_colunas(tmp_path):
    df = load_runs(tmp_path)
    assert len(df) == 0 and "macro_dice" in df.columns

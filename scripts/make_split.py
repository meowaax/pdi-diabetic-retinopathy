"""Gera configs/split.csv: treino/validação/teste por imagem original e dobras de validação cruzada.

Colunas: subset (divisão de desenvolvimento: train/val/test) e fold (0-4 nas 54 imagens oficiais de
treino; -1 no teste). A divisão de desenvolvimento serve à sensibilidade do CLAHE (#24) e ao ajuste de
hiperparâmetros (#13); as dobras servem à comparação final E1-E4 (#17).

- teste = divisão oficial do IDRiD (27 imagens);
- validação = 12 das 54 imagens oficiais de treino, estratificadas por (SE disponível) x (carga de lesão);
- IDRiD_43 (sem máscara de HE) fica sempre no treino, para a validação ter anotação completa de MA/HE/EX.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

SEED = 42
N_VAL = 12

avail = pd.read_csv("results/mask_availability.csv")
img = pd.read_csv("results/eda/per_image.csv")
burden = img.groupby("image_id").frac_fov.sum().rename("burden")  # soma das frações das classes anotadas

train = avail[avail.split == "train"].merge(burden, on="image_id")
elig = train[train.HE == 1].copy()                      # exclui IDRiD_43 da validação
elig["burden_bin"] = elig.groupby("SE").burden.transform(lambda s: pd.qcut(s, 2, labels=False))
elig["stratum"] = elig.SE.astype(str) + "_" + elig.burden_bin.astype(str)

rng = np.random.default_rng(SEED)
props = elig.stratum.value_counts(normalize=True) * N_VAL
n_per = props.round().astype(int)
n_per.iloc[0] += N_VAL - n_per.sum()                    # ajusta arredondamento
val_ids = []
for st, n in n_per.items():
    ids = elig[elig.stratum == st].image_id.to_numpy()
    val_ids += list(rng.choice(ids, size=n, replace=False))

out = avail[["image_id", "split", "MA", "HE", "EX", "SE"]].copy()
out["subset"] = np.where(out.split == "test", "test", np.where(out.image_id.isin(val_ids), "val", "train"))
out = out.drop(columns="split").sort_values("image_id")
# dobras (5-fold) sobre as 54 imagens oficiais de treino, estratificadas por SE x carga de lesão
N_FOLDS = 5
rng_f = np.random.default_rng(SEED + 1)
dev = train.copy()
dev["burden_bin"] = dev.groupby("SE").burden.transform(lambda s: pd.qcut(s, 2, labels=False))
fold_of, counter = {}, 0
for _, grp in dev.groupby(["SE", "burden_bin"]):
    for i in rng_f.permutation(grp.image_id.to_numpy()):
        fold_of[i] = counter % N_FOLDS
        counter += 1
out["fold"] = out.image_id.map(fold_of).fillna(-1).astype(int)

Path("configs").mkdir(exist_ok=True)
out.to_csv("configs/split.csv", index=False)

# verificações
assert out.image_id.is_unique
c = out.subset.value_counts().to_dict()
assert c == {"train": 42, "val": 12, "test": 27}, c
assert set(out[out.subset == "test"].image_id) == set(avail[avail.split == "test"].image_id)
assert out[out.image_id == "IDRiD_43"].subset.item() == "train"
assert (out[out.subset == "test"].fold == -1).all() and sorted(out[out.fold >= 0].fold.unique()) == list(range(N_FOLDS))
print("\ndobras:")
print(out[out.fold >= 0].groupby("fold")[["MA", "HE", "EX", "SE"]].sum().assign(n=out[out.fold >= 0].fold.value_counts().sort_index()))
print(out.groupby("subset")[["MA", "HE", "EX", "SE"]].sum().assign(n=out.subset.value_counts()))
print("\nval:", sorted(out[out.subset == "val"].image_id))
b = out.merge(burden, on="image_id")
print("\ncarga média de lesão (% FOV somado):")
print((b.groupby("subset").burden.mean() * 100).round(2))

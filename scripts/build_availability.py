"""Gera results/mask_availability.csv e confere contagens e dimensões."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data import CLASSES, ROOT, availability_table, image_path, mask_path

EXPECTED = {
    "train": {"images": 54, "MA": 54, "HE": 53, "EX": 54, "SE": 26},
    "test": {"images": 27, "MA": 27, "HE": 27, "EX": 27, "SE": 14},
}

df = availability_table()
out = Path("results/mask_availability.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

ok = True
for split, exp in EXPECTED.items():
    sub = df[df.split == split]
    got = {"images": len(sub), **{c: int(sub[c].sum()) for c in CLASSES}}
    status = "OK" if got == exp else "DIVERGE"
    ok &= got == exp
    print(f"{split:5} {got}  [{status}]")

# arquivos de máscara sem imagem correspondente
for split in ("train", "test"):
    ids = set(df[df.split == split].image_id)
    for cls in CLASSES:
        folder = mask_path("X", cls, split).parent
        extra = {p.name.rsplit("_", 1)[0] for p in folder.glob("*.tif")} - ids
        if extra:
            ok = False
            print("máscaras órfãs", split, cls, sorted(extra))

# dimensões: todas as máscaras devem ter o mesmo tamanho da imagem
bad = 0
for r in df.itertuples():
    h, w = cv2.imread(str(image_path(r.image_id, r.split)), cv2.IMREAD_COLOR).shape[:2]
    for cls in CLASSES:
        if getattr(r, cls):
            mh, mw = cv2.imread(str(mask_path(r.image_id, cls, r.split)), cv2.IMREAD_UNCHANGED).shape[:2]
            if (mh, mw) != (h, w):
                bad += 1
                print("tamanho diferente:", r.image_id, cls, (mh, mw), (h, w))
ok &= bad == 0

print("\nausentes:")
for cls in CLASSES:
    miss = df[df[cls] == 0]
    print(f"  {cls}: {len(miss)} -> train {sum(miss.split=='train')}, test {sum(miss.split=='test')}")
print("\nCSV:", out)
sys.exit(0 if ok else 1)

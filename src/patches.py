"""Cache em disco e Dataset de patches 512x512 do IDRiD.

Cache (data/cache/, não versionado), por imagem:
  {id}_img.npy   uint8 (H, W, 3)  imagem RGB
  {id}_lab.npy   uint8 (H, W)     bits 0-3 = MA,HE,EX,SE (1 = lesão); bit 4 = campo de visão (FOV)
  {id}_meta.npz  valid (4,) uint8, pts_MA/HE/EX/SE (N, 2) int32 com coordenadas (y, x) de pixels de lesão
Os .npy são abertos com mmap, então só os patches pedidos vão para a memória.

Treino ('train'): patches aleatórios; com probabilidade `lesion_prob` o patch é centrado (com jitter)
num pixel de lesão de uma classe sorteada entre as classes com anotação e lesão; caso contrário, uniforme
dentro do FOV. Avaliação ('eval'): grade determinística com `stride`, descartando patches com FOV < `min_fov`.
Augmentation e pré-processamento entram como funções (ver #8 e #6) e atuam no patch, nessa ordem.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CLASSES, load_image, load_masks
from .split import source_split

CACHE = Path("data/cache")
FOV_BIT = 4
MAX_PTS = 20000
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_fov(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fov = (gray > 15).astype(np.uint8)
    return cv2.morphologyEx(fov, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)).astype(bool)


def build_cache(image_ids, cache: Path = CACHE, seed: int = 42, overwrite: bool = False) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for image_id in image_ids:
        if not overwrite and (cache / f"{image_id}_meta.npz").exists():
            continue
        split = source_split(image_id)
        img = load_image(image_id, split)
        masks, valid = load_masks(image_id, split, img.shape[:2])
        lab = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = {}
        for k, c in enumerate(CLASSES):
            lab |= (masks[k] > 0).astype(np.uint8) << k
            ys, xs = np.nonzero(masks[k])
            if len(ys) > MAX_PTS:
                sel = rng.choice(len(ys), MAX_PTS, replace=False)
                ys, xs = ys[sel], xs[sel]
            pts[f"pts_{c}"] = np.stack([ys, xs], 1).astype(np.int32)
        lab |= compute_fov(img).astype(np.uint8) << FOV_BIT
        np.save(cache / f"{image_id}_img.npy", img)
        np.save(cache / f"{image_id}_lab.npy", lab)
        np.savez(cache / f"{image_id}_meta.npz", valid=valid, **pts)


def grid_positions(h: int, w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    """Cantos superiores esquerdos da grade; a última linha/coluna é encostada na borda."""
    def axis(n):
        if n <= patch:
            return [0]
        p = list(range(0, n - patch + 1, stride))
        if p[-1] != n - patch:
            p.append(n - patch)
        return p
    return [(y, x) for y in axis(h) for x in axis(w)]


def stitch(logits: np.ndarray, positions, shape: tuple[int, int]) -> np.ndarray:
    """Une logits (N, C, P, P) de patches sobrepostos em (C, H, W) pela média nas sobreposições."""
    n, c, p, _ = logits.shape
    acc = np.zeros((c, *shape), dtype=np.float32)
    cnt = np.zeros(shape, dtype=np.float32)
    for lg, (y, x) in zip(logits, positions):
        acc[:, y:y + p, x:x + p] += lg
        cnt[y:y + p, x:x + p] += 1
    return acc / np.maximum(cnt, 1)


def _unpack(lab: np.ndarray):
    masks = np.stack([(lab >> k) & 1 for k in range(len(CLASSES))], axis=-1)  # (h, w, 4)
    fov = (lab >> FOV_BIT) & 1
    return masks, fov


class IDRiDPatches(Dataset):
    def __init__(self, image_ids, mode="train", patch=512, stride=256, patches_per_image=24,
                 lesion_prob=0.7, jitter=192, min_fov=0.2, transform=None, preprocess=None,
                 normalize=True, seed=42, cache: Path = CACHE):
        assert mode in ("train", "eval")
        self.ids, self.mode, self.patch, self.stride = list(image_ids), mode, patch, stride
        self.ppi, self.lesion_prob, self.jitter, self.min_fov = patches_per_image, lesion_prob, jitter, min_fov
        self.transform, self.preprocess, self.normalize = transform, preprocess, normalize
        self.seed, self.epoch, self.cache = seed, 0, cache
        build_cache(self.ids, cache)
        self._meta = {}
        for i in self.ids:
            z = np.load(cache / f"{i}_meta.npz")
            self._meta[i] = {k: z[k] for k in z.files}
        self._lab = {i: np.load(cache / f"{i}_lab.npy", mmap_mode="r") for i in self.ids}
        self._img = {i: np.load(cache / f"{i}_img.npy", mmap_mode="r") for i in self.ids}
        self.shapes = {i: self._img[i].shape[:2] for i in self.ids}
        if mode == "eval":
            self.items = []
            for i in self.ids:
                h, w = self.shapes[i]
                for y, x in grid_positions(h, w, patch, stride):
                    fov = (np.asarray(self._lab[i][y:y + patch, x:x + patch]) >> FOV_BIT) & 1
                    if fov.mean() >= min_fov:
                        self.items.append((i, y, x))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.items) if self.mode == "eval" else len(self.ids) * self.ppi

    def _sample_train(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        image_id = self.ids[idx % len(self.ids)]
        h, w = self.shapes[image_id]
        P, meta = self.patch, self._meta[image_id]
        if rng.random() < self.lesion_prob:
            cands = [c for k, c in enumerate(CLASSES) if meta["valid"][k] and len(meta[f"pts_{c}"])]
            if cands:
                pts = meta[f"pts_{rng.choice(cands)}"]
                cy, cx = pts[rng.integers(len(pts))]
                cy += rng.integers(-self.jitter, self.jitter + 1)
                cx += rng.integers(-self.jitter, self.jitter + 1)
                y = int(np.clip(cy - P // 2, 0, h - P))
                x = int(np.clip(cx - P // 2, 0, w - P))
                return image_id, y, x
        for _ in range(50):  # uniforme dentro do FOV
            y, x = int(rng.integers(0, h - P + 1)), int(rng.integers(0, w - P + 1))
            if ((np.asarray(self._lab[image_id][y:y + P, x:x + P]) >> FOV_BIT) & 1).mean() >= self.min_fov:
                break
        return image_id, y, x

    def __getitem__(self, idx):
        image_id, y, x = self.items[idx] if self.mode == "eval" else self._sample_train(idx)
        P = self.patch
        img = np.ascontiguousarray(self._img[image_id][y:y + P, x:x + P])
        masks, _ = _unpack(np.asarray(self._lab[image_id][y:y + P, x:x + P]))
        masks = np.ascontiguousarray(masks)
        if self.transform is not None:
            out = self.transform(image=img, mask=masks)
            img, masks = out["image"], out["mask"]
        if self.preprocess is not None:
            img = self.preprocess(img)
        img = img.astype(np.float32) / 255.0
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        valid = self._meta[image_id]["valid"].astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            "mask": torch.from_numpy(np.ascontiguousarray(masks.transpose(2, 0, 1))).float(),
            "valid": torch.from_numpy(valid),
            "image_id": image_id, "y": y, "x": x,
        }

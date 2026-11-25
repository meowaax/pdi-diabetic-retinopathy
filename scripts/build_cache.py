"""Gera data/cache/ para as 81 imagens (rodar uma vez; ~4 GB em disco)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import availability_table
from src.patches import build_cache

ids = availability_table().image_id.tolist()
build_cache(ids)
print("cache pronto:", len(ids), "imagens")

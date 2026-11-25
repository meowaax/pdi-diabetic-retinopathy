import numpy as np
import pytest

from src.preprocess import preprocess, preprocess_clahe, preprocess_green, preprocess_rgb


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


def test_rgb_e_identidade(img):
    assert np.array_equal(preprocess(img, "rgb"), img)
    assert preprocess_rgb(img) is img


def test_formato_tipo_e_faixa_preservados_em_todos_os_modos(img):
    for mode in ("rgb", "clahe", "green", "clahe_green"):
        out = preprocess(img, mode)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255


def test_canal_verde_replicado_e_igual_nos_3_canais(img):
    out = preprocess(img, "green")
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert np.array_equal(out[..., 0], img[..., 1])


def test_clahe_aumenta_contraste_em_imagem_de_baixo_contraste():
    low = np.full((64, 64, 3), 128, dtype=np.uint8)
    low[20:40, 20:40] = 140  # região levemente mais clara, baixo contraste
    out = preprocess_clahe(low, clip_limit=3.0)
    l_in = low[..., 1].astype(np.float32).std()
    l_out = out[..., 1].astype(np.float32).std()
    assert l_out > l_in


def test_clahe_preserva_cor_em_tons_de_cinza(img):
    """Em imagem sem cor, CLAHE no canal L não deve introduzir tonalidade (R=G=B)."""
    gray = np.repeat(img[..., :1], 3, axis=2)
    out = preprocess_clahe(gray)
    assert np.abs(out[..., 0].astype(int) - out[..., 1].astype(int)).max() <= 2
    assert np.abs(out[..., 1].astype(int) - out[..., 2].astype(int)).max() <= 2


def test_clahe_green_e_a_composicao_na_ordem_certa(img):
    assert np.array_equal(preprocess(img, "clahe_green"), preprocess_green(preprocess_clahe(img)))


def test_parametros_do_clahe_sao_repassados():
    # tile quase uniforme com poucos pixels de pico: o clipping trunca o histograma de forma
    # diferente conforme clip_limit (ruído puro ou gradiente suave saturam igual em qualquer valor)
    tile = np.full((8, 8), 100, dtype=np.uint8)
    tile[0, :2] = 250
    img = np.repeat(np.tile(tile, (8, 8))[..., None], 3, axis=2)
    a = preprocess(img, "clahe", clip_limit=1.0, tile_grid_size=(8, 8))
    b = preprocess(img, "clahe", clip_limit=40.0, tile_grid_size=(8, 8))
    assert not np.array_equal(a, b)


def test_modo_invalido_levanta_erro(img):
    with pytest.raises(ValueError):
        preprocess(img, "bgr")

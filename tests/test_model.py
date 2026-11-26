import torch

from src.model import build_model, predict_proba
from src.seed import get_device, set_seed


def test_forward_backward_em_patch_512_sem_pesos_pretreinados():
    set_seed(0)
    device = get_device()
    model = build_model(encoder_weights=None).to(device)
    x = torch.randn(2, 3, 512, 512, device=device)
    y = torch.randint(0, 2, (2, 4, 512, 512), device=device).float()

    logits = model(x)
    assert logits.shape == (2, 4, 512, 512)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    # decoder.center não participa do forward da UnetPlusPlus com este encoder (módulo não usado
    # pela arquitetura para VGG16); todo o resto da rede deve receber gradiente.
    sem_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert all(n.startswith("decoder.center") for n in sem_grad), sem_grad
    assert any(n.startswith("encoder") for n, p in model.named_parameters() if p.grad is not None)
    assert any(n.startswith("segmentation_head") for n, p in model.named_parameters() if p.grad is not None)


def test_predict_proba_fica_em_0_1():
    model = build_model(encoder_weights=None)
    x = torch.randn(1, 3, 64, 64)
    proba = predict_proba(model, x)
    assert proba.shape == (1, 4, 64, 64)
    assert proba.min() >= 0 and proba.max() <= 1


def test_batch_1_a_8_cabe_em_512x512_na_gpu_disponivel():
    """Cada batch é testado isoladamente, com no_grad e liberação explícita de memória entre eles:
    sem isso, as ativações retidas para o backward (que este teste não faz) se acumulam de um
    batch para o outro e podem estourar a VRAM mesmo quando cada batch, por si só, caberia."""
    device = get_device()
    model = build_model(encoder_weights=None).to(device)
    for b in (1, 4, 8):
        with torch.no_grad():
            x = torch.randn(b, 3, 512, 512, device=device)
            logits = model(x)
            assert logits.shape == (b, 4, 512, 512)
        del logits, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

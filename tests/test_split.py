from src.split import ids_for, load_split, source_split


def test_sem_sobreposicao_e_contagens():
    tr, va, te = ids_for("train"), ids_for("val"), ids_for("test")
    assert (len(tr), len(va), len(te)) == (42, 12, 27)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_teste_e_o_oficial():
    assert all(source_split(i) == "test" for i in ids_for("test"))
    assert all(source_split(i) == "train" for i in ids_for("train") + ids_for("val"))


def test_validacao_tem_anotacao_completa_de_ma_he_ex():
    df = load_split()
    v = df[df.subset == "val"]
    assert v[["MA", "HE", "EX"]].all().all()
    assert v.SE.sum() == 6

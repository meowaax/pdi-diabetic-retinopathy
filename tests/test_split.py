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


def test_dobras_cobrem_o_treino_oficial_sem_repetir_e_sem_teste():
    from src.split import fold_ids

    todas_val = []
    for k in range(5):
        tr, va = fold_ids(k)
        assert not set(tr) & set(va)
        assert not (set(tr) | set(va)) & set(ids_for("test"))
        assert len(tr) + len(va) == 54
        todas_val += va
    assert sorted(todas_val) == sorted(ids_for("train") + ids_for("val"))
    assert len(set(todas_val)) == 54

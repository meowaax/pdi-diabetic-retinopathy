"""Gera notebooks/01_eda.ipynb (leitura dos resultados de results/eda/)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 01. Análise exploratória do IDRiD (segmentação)\n\nGerado a partir de `scripts/eda_stats.py` e `scripts/eda_figures.py`. "
       "Frações calculadas sobre o campo de visão (FOV) da retina."),
    code("import pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.read_csv('../results/eda/resumo_por_classe.csv', index_col=0)"),
    md("## Fração de pixels por classe\nMA ocupa ~0,15% do FOV; HE e EX ~1,3-1,5%; SE ~0,5%. "
       "Por pixel, HE (46%) e EX (41%) dominam o total de pixels de lesão, enquanto MA representa só ~5%."),
    code("display(Image('../results/eda/frac_pixels_por_classe.png'))"),
    md("## Tamanho e número de lesões\nMA e EX são numerosos e pequenos (mediana ~220-230 px); HE e SE são maiores e menos frequentes. "
       "SE tem só 150 componentes em 40 imagens."),
    code("display(Image('../results/eda/tamanho_componentes.png'))\ndisplay(Image('../results/eda/componentes_por_imagem.png'))"),
    md("## Exemplos com máscaras sobrepostas\nInclui imagens sem a máscara de HE/SE (IDRiD_43) para lembrar que ausência de arquivo não é ausência de lesão."),
    code("display(Image('../results/eda/overlay_exemplos.png'))"),
    md("## Observações para a modelagem\n"
       "- Desbalanceamento forte: MA tem ~10x menos pixels que HE/EX. Considerar pesos por classe ou Dice na loss.\n"
       "- MA exige resolução alta: lesões de 13 a ~2200 px (mediana ~230 px) na resolução original de 2848×4288; reduzir demais as apaga.\n"
       "- Há imagens com lesões enormes (ex.: HE em IDRiD_17, 1,26 M px): patches de 512×512 podem cortar essas regiões.\n"
       "- SE existe em só 40 imagens, então a perda mascarada é essencial.\n"
       "- Dado atípico: `IDRiD_81_EX.tif` é BGRA (canal alfa = 255 em todos os pixels, lesão no canal vermelho). "
       "O carregador descarta o alfa; sem isso a máscara viraria a imagem inteira."),
]
nbf.write(nb, "notebooks/01_eda.ipynb")

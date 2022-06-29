from pathlib import Path
from glob import glob

import pandas as pd

df_comp = pd.read_csv('pellets-RGB-classes.csv', index_col = 0)
df_visual = pd.read_csv('pellets-visual-classes.csv', index_col = 0)


fnames = glob("01-masks/*.npy")

comp_index = df_comp.index
visual_index = df_visual.index
bad_pellets = comp_index.difference(visual_index)

dataset = {}
for fname in fnames:
    pellet = Path(fname).stem
    if pellet in bad_pellets:
        continue
    #get the yellowness class of the pellet for the rotine and for the visual classification
    class_comp = df_comp.loc[pellet][3]
    class_visual = df_visual.loc[pellet][5]
    #compare the yellowness
    if class_comp == class_visual:
        dataset.update({pellet: ("Mesma Classe", class_comp)})
    else:
        dataset.update({pellet: ("Classes Diferentes", "")})

df = pd.DataFrame(dataset, index = ("Comparação das Metodologias", "Grau de Amarelamento")).T
df.to_csv("comparison_methodologies.csv")


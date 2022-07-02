from pathlib import Path
from glob import glob

import pandas as pd

df_rgb_blue = pd.read_csv('pellets-RGB-classes-onlyblue.csv', index_col = 0)
df_rgb_br = pd.read_csv("pellets-RGB-classes-br.csv", index_col = 0)
df_visual = pd.read_csv('pellets-visual-classes.csv', index_col = 0)


fnames = glob("01-masks/*.npy")

rgb_index = df_rgb_blue.index
visual_index = df_visual.index
bad_pellets = rgb_index.difference(visual_index)

dataset = {}
for fname in fnames:
    pellet = Path(fname).stem
    if pellet in bad_pellets:
        continue
    #get the yellowness class of the pellet for the rotine and for the visual classification
    class_blue = df_rgb_blue.loc[pellet][3]
    class_br = df_rgb_br.loc[pellet][3]
    class_visual = df_visual.loc[pellet][5]
    
    #compare the yellowness
    if class_blue == class_visual and class_br == class_visual:
        dataset.update({pellet: (class_visual, class_blue, class_br, "Mesma Classe", "Mesma Classe")})
    elif class_blue == class_visual and class_br != class_visual:
        dataset.update({pellet: (class_visual, class_blue, class_br, "Mesma Classe", "Classes Diferentes")})
    elif class_blue != class_visual and class_br == class_visual:
        dataset.update({pellet: (class_visual, class_blue, class_br, "Classes Diferentes", "Mesma Classe")})
    else:
        dataset.update({pellet: (class_visual, class_blue, class_br, "Classes Diferentes", "Classes Diferentes")})


df = pd.DataFrame(dataset, index = ("Classificação Visual",                          
                                    "Classificação RGB - B",
                                    "Classificação RGB - R/B",
                                    "Comparação Visual e B",
                                    "Comparação Visual e R/B"
                                   )).T
df.to_csv("comparison_methodologies.csv")


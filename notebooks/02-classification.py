"""Classification of yellowness from the images."""

import numpy as np
import pandas as pd

df = pd.read_csv("RGB_values.csv", index_col=0)
df2 = pd.read_csv("RGB_values.csv", index_col=0)


#onlyblue

blue = {
    "nulo":       (df["B"] >= 160),
    "baixo":      (df["B"] <  160) & (df["B"] >  55),
    "moderado":   (df["B"] <= 55)  & (df["B"] >  10),
    "alto":       (df["B"] <= 10),
    "muito alto": (df["B"] <= 10)
}

# create a new column and use np.select to assign values to it using our lists as arguments
df["Grau de Amarelamento"] = np.select(blue.values(), blue.keys())
df.to_csv("pellets-RGB-classes-onlyblue.csv")

#blue+ b/r

blue = {
    "nulo":       (df["B"] >= 160),
    "baixo":      (df["B"] <  160) & (df["B"] >  55),
    "moderado":   (df["B"] <= 55)  & (df["B"] >  10),
    "alto":       (df["B"] <= 10) & (df['B']/df['R'] < 0.04),
    "muito alto": (df["B"] <= 10)
}

# create a new column and use np.select to assign values to it using our lists as arguments
df2["Grau de Amarelamento"] = np.select(blue.values(), blue.keys())
df2.to_csv("pellets-RGB-classes-br.csv")
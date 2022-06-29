from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

# classification of yellowness from the images

df = pd.read_csv('RGB_values.csv', index_col = 0)


# create a list of our conditions
conditions = [
    (df['B'] >= 160) & (df['G']/df['B'] <= 1.5) & (df['G']/df['B'] > 0),
    (df['B'] < 160) & (df['B'] >= 55),
    (df['B'] < 55) & (df['B'] > 10),
    (df['B'] <= 10) & (df['G'] >= 115),
    (df['B'] <= 10) & (df['G'] < 115)
    ]

# create a list of the values we want to assign for each condition
values = ['nulo', 'baixo', 'moderado', 'alto', 'muito alto']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Grau de Amarelamento'] = np.select(conditions, values)
df.to_csv("pellets-RGB-classes.csv")
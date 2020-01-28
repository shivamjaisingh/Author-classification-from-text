import numpy as np
import pandas as pd
import scipy
from sklearn.utils import shuffle

df = pd.read_csv('~/Desktop/file.csv')
print(len(df))
df = shuffle(df)
df.reset_index(inplace=True, drop=True)
print(df.head(100))


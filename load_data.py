import numpy as np
import pickle
import pandas as pd

with open ('../data/training/x.pkl', 'rb') as fp:
    X = pickle.load(fp)

y = pd.read_excel('../data/training/y.xlsx')

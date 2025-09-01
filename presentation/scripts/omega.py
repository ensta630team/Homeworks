import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

from source.fit.inference.test_chow import test_chow
from statsmodels.tsa.api import VAR

from source.data.loaders import create_dataset
from source.models.ivar import IVAR
from source.data.transform import create_var_dataset
from source.display.hw import hw2 as plots
from source.display.hw.hw1 import save_figure

path = './data/base_25.xls'
dataset = create_dataset(path, problem=5)


i_reshaped = dataset['i'][12:]
X = np.vstack([dataset['pi'], dataset['y'], i_reshaped]).T
X = np.vstack([dataset['pi'], dataset['y'], i_reshaped]).T

df = pd.DataFrame(X, columns=['pi', 'y', 'i'])
model = VAR(df)
results = model.fit(2)
print(results.sigma_u)
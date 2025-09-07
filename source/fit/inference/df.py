import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import mackinnonp
#from statsmodels.tsa.stattools import adfuller

def test_df(serie_array: np.ndarray, estacionalizar = "no") -> pd.DataFrame:
  if estacionalizar == "no":
    ## ------------ Trabando las series ------------
    serie = serie_array
    n = len(serie)
    yt = serie[1:n]
    yt1 = serie[0:n-1]
    # Delta y
    dy = yt-yt1
    # tendencia
    t = list(range(1,n))
    # vector Y y matriz X de regresores
    Y = dy
    X = np.column_stack((np.ones(n-1), yt1, t))

    ## ------------ Estadistico Observado ------------
    # Estimacion por OLS
    delta = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
    # Calculo estimador varianza
    residuos = Y - X @ delta
    residuos2 = residuos ** 2
    suma_residuos2 = np.sum(residuos2)
    omega = suma_residuos2 / (n-3) * np.linalg.inv((X.T @ X))
    # Estadistico Observado
    R = np.zeros((1,3))
    R[0, 1] = 1
    estadistico = delta[1] / np.sqrt((R @ omega @ R.T))
    estadistico = estadistico.squeeze()

    ## ------------ valor-p ------------
    # Ocupo libreria
    valorp = mackinnonp(estadistico, regression='ct')

    ## ------------ comprobar ------------
    #stat, pval = adfuller(serie_array, regression='ct', autolag='AIC')[:2]

    #print("Test DF")
    result = {
        'statistic': estadistico,
        'p-value': valorp,
        #'compro1': stat,
        #'compro2': pval
    }

    return pd.DataFrame(result, index=[0])
  else:
    ## ------------ Trabando las series ------------
    serie = serie_array
    n = len(serie)
    yt = serie[1:n]
    yt1 = serie[0:n-1]
    # Delta y
    dy = yt-yt1
    # tendencia
    t = list(range(1,n))
    print(len(dy))
    print(len(t))
    # Dummies por mes
    meses = np.arange(n-1) % 12
    meses = np.eye(12)[meses]
    meses = meses[:, 0:11]
    # vector Y y matriz X de regresores
    Y = dy
    X = np.column_stack((np.ones(n-1), yt1, t, meses))

    ## ------------ Estadistico Observado ------------
    # Estimacion por OLS
    delta = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
    # Calculo estimador varianza
    residuos = Y - X @ delta
    residuos2 = residuos ** 2
    suma_residuos2 = np.sum(residuos2)
    omega = suma_residuos2 / (n-14) * np.linalg.inv((X.T @ X))
    # Estadistico Observado
    R = np.zeros((1,14))
    R[0, 1] = 1
    estadistico = delta[1] / np.sqrt((R @ omega @ R.T))
    estadistico = estadistico.squeeze()

    ## ------------ valor-p ------------
    # Ocupo libreria
    valorp = mackinnonp(estadistico, regression='ct')

    #print("Test DF")
    result = {
        'statistic': estadistico,
        'p-value': valorp,
        #'compro1': stat,
        #'compro2': pval
    }

    return pd.DataFrame(result, index=[0])

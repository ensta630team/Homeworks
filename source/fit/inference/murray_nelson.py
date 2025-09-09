import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import mackinnonp
#from statsmodels.tsa.stattools import adfuller

def test_murray_nelson(serie_array: np.ndarray, periodo:None,  persistent:None = None,  k:None = 0) -> pd.DataFrame:
    periodos = [x - k for x in periodo]
    if persistent is not None:
      persistente = persistent - k
    ## ------------ Trabando las series ------------
    serie = serie_array
    n = len(serie)
    yt = serie[1:n]
    yt1 = serie[0:n-1]
    # Delta y
    dy = yt-yt1
    # tendencia
    t = list(range(1,n))
    # Delta y Lageados
    if k == 0:
      lags_t = t
    else:
      lags_t = np.zeros((n-1-k,k+1))
      for i in range(0,k):
        lags_t[:,i] = dy[k-1-i:n-2-i]
        #dy[2:n-2]
        #dy[1:n-3]
        #dy[0:n-4]
      dy = dy[k:n-1]
      yt1 = yt1[k:n-1]
      t = t[k:n-1]
      lags_t[:,k] = t
    # Dummies por mes
    meses = np.arange(n-1-k) % 12
    meses = np.eye(12)[meses]
    meses = meses[:, 0:11]
    # vector Y y matriz X de regresores
    Y = dy
    # Dummies outliers
    if periodo is None and persistent is None:
      X = np.column_stack((np.ones(n-1-k), yt1, lags_t, meses))
    elif periodo is not None and persistent is None:
      dummies = np.zeros((n-1-k,len(periodos)))
      for i,x in enumerate(periodos):
        dummies[x,i] = 1
        X = np.column_stack((np.ones(n-1-k), yt1, lags_t, dummies, meses))
    elif periodo is None and persistent is not None:
      dummy_persistente = np.zeros((n-1-k,))
      dummy_persistente[persistente:] = 1
      X = np.column_stack((np.ones(n-1-k), yt1, lags_t, dummy_persistente, meses))
    else:
      dummy_persistente = np.zeros((n-1-k,))
      dummy_persistente[persistente:] = 1
      dummies = np.zeros((n-1-k,len(periodos)))
      for i,x in enumerate(periodos):
        dummies[x,i] = 1
        X = np.column_stack((np.ones(n-1-k), yt1, lags_t, dummies,dummy_persistente, meses))


    ## ------------ Estadistico Observado ------------
    if persistent is None:
      klib = 14 + len(periodos) + k
    else:
      klib = 14 + len(periodos) + 1 + k
    # Estimacion por OLS
    delta = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
    # Calculo estimador varianza
    residuos = Y - X @ delta
    residuos2 = residuos ** 2
    suma_residuos2 = np.sum(residuos2)
    omega = suma_residuos2 / (n-3) * np.linalg.inv((X.T @ X))
    # Estadistico Observado
    R = np.zeros((1,klib))
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
        'model': 'murray_nelson'
        #'compro2': pval
    }

    return pd.DataFrame(result, index=[0])

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import mackinnonp
#from statsmodels.tsa.stattools import adfuller

def test_murray_nelson(serie_array: np.ndarray, periodos:None,  persistente:None = None) -> pd.DataFrame:
    ## ------------ Trabando las series ------------
    serie = serie_array
    n = len(serie)
    yt = serie[1:n]
    yt1 = serie[0:n-1]
    # Delta y
    dy = yt-yt1
    # tendencia
    t = list(range(1,n))
    # Dummies por mes
    meses = np.arange(n-1) % 12
    meses = np.eye(12)[meses]
    meses = meses[:, 0:11]
    # vector Y y matriz X de regresores
    Y = dy
    # Dummies outliers
    if periodos is None and persistente is None:
      X = np.column_stack((np.ones(n-1), yt1, t, meses))
    elif periodos is not None and persistente is None:
      dummies = np.zeros((n-1,len(periodos)))
      for i,x in enumerate(periodos):
        dummies[x,i] = 1
        X = np.column_stack((np.ones(n-1), yt1, t, dummies, meses))
    elif periodos is None and persistente is not None:
      dummy_persistente = np.zeros((n-1,))
      dummy_persistente[persistente:] = 1
      X = np.column_stack((np.ones(n-1), yt1, t, dummy_persistente, meses))
    else:
      dummy_persistente = np.zeros((n-1,))
      dummy_persistente[persistente:] = 1
      dummies = np.zeros((n-1,len(periodos)))
      for i,x in enumerate(periodos):
        dummies[x,i] = 1
        X = np.column_stack((np.ones(n-1), yt1, t, dummies,dummy_persistente, meses))


    ## ------------ Estadistico Observado ------------
    # Estimacion por OLS
    delta = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
    # Calculo estimador varianza
    residuos = Y - X @ delta
    residuos2 = residuos ** 2
    suma_residuos2 = np.sum(residuos2)
    omega = suma_residuos2 / (n-3) * np.linalg.inv((X.T @ X))
    # Estadistico Observado
    if persistente is None:
        k = 14 + len(periodos)
    else:
        k = 14 + len(periodos) + 1
    R = np.zeros((1,k))
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

import pandas as pd
import numpy as np

import statsmodels.api as sm

def test_ljungbox(serie_array: np.ndarray, lags: int) -> pd.DataFrame:
    """
    Realiza el test de Ljung-Box en un array de NumPy.

    Args:
        serie_array (np.ndarray): Un array de NumPy con la serie de tiempo.
        lags (int): El número de rezagos a testear.

    Returns:
        pd.DataFrame: Un DataFrame con las métricas del test para cada rezago.
    """
    print('Test Ljung-Box')

    # Convertir el array de NumPy a una serie de pandas
    serie = pd.Series(serie_array)
    
    # Realizar el test de Ljung-Box
    out = sm.stats.diagnostic.acorr_ljungbox(serie, lags=[lags])

    result = {
        'statistic': out.iloc[0]['lb_stat'],
        'p-value': out.iloc[0]['lb_pvalue']

    }

    return pd.DataFrame(result, index=[0])
import numpy as np
from statsmodels.tsa.api import VAR
import warnings
import pandas as pd

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings("ignore")

def hannan_rissanen(datos, max_rezagos=15, verbose=True):
    """
    Evalúa diferentes órdenes de rezago (p) para un modelo VAR y devuelve
    una tabla con los criterios de información AIC, BIC y HQIC.

    Args:
        datos (np.ndarray): Un array de NumPy con las series de tiempo,
                            donde las columnas son las variables y las filas son
                            las observaciones.
        max_rezagos (int): El número máximo de rezagos a evaluar.

    Returns:
        pandas.DataFrame: Una tabla con los valores de AIC, BIC y HQIC para
                          cada orden de rezago p.
    """
    resultados = []
    print("Evaluando órdenes de rezago del 1 al {}...".format(max_rezagos))
    
    pbar = range(1, max_rezagos + 1)
    for p in pbar:
        model = VAR(endog=datos)
        results = model.fit(maxlags=p)
        
        # Recolectar los criterios de información
        aic = results.aic
        bic = results.bic
        hqic = results.hqic
        resultados.append({'p': p, 'AIC': aic, 'BIC': bic, 'HQIC': hqic})

    # Convertir la lista de resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados).set_index('p')
    print("Evaluación completada.")
    return df_resultados
import numpy as np
from statsmodels.tsa.api import VAR
import warnings
import pandas as pd

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings("ignore")

import numpy as np
from statsmodels.tsa.api import VAR
import warnings
import pandas as pd

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings("ignore")

def hannan_rissanen_con_ljungbox(datos, max_rezagos=15, verbose=True):
    """
    Evalúa diferentes órdenes de rezago (p) para un modelo VAR y devuelve
    una tabla con los criterios de información AIC, BIC, HQIC y el p-valor
    del test de Ljung-Box para la autocorrelación de los residuos.

    Args:
        datos (np.ndarray): Un array de NumPy con las series de tiempo,
                            donde las columnas son las variables y las filas son
                            las observaciones.
        max_rezagos (int): El número máximo de rezagos a evaluar.

    Returns:
        pandas.DataFrame: Una tabla con los valores de AIC, BIC, HQIC y el
                          p-valor del test de Ljung-Box para cada orden de rezago p.
    """
    resultados = []
    if verbose:
        print(f"Evaluando órdenes de rezago del 1 al {max_rezagos}...")

    pbar = range(1, max_rezagos + 1)
    for p in pbar:
        model = VAR(endog=datos)
        results = model.fit(maxlags=p, ic=None)

        # Criterios de información
        aic = results.aic
        bic = results.bic
        hqic = results.hqic

        # ======================================================================
        # CORRECCIÓN AQUÍ: Usar nlags=2*p en lugar de nlags=p
        # El número de rezagos del test (nlags) debe ser > que los del modelo (p).
        # ======================================================================
        ljung_box_results = results.test_whiteness(nlags=2*p, signif=0.05)
        lb_pvalue = ljung_box_results.pvalue

        # Almacenar resultados
        resultados.append({
            'p': p,
            'AIC': aic,
            'BIC': bic,
            'HQIC': hqic,
            'Ljung-Box p-valor': lb_pvalue
        })

    # Convertir la lista de resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados).set_index('p')
    if verbose:
        print("Evaluación completada.")
    return df_resultados
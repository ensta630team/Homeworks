import numpy as np
import warnings
import pandas as pd
import statsmodels.api as sm
import os

from source.models.ols import OLS

from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox


# Suprimir advertencias para una salida más limpia
warnings.filterwarnings("ignore")



def hannan_rissanen(datos, break_point=50, max_rezagos=15, verbose=True, model_type='var', inp_fn=None):
    """
    Evalúa diferentes órdenes de rezago (p) para un modelo VAR y devuelve
    una tabla con los criterios de información AIC, BIC y HQIC.

    Args:
        datos (np.ndarray): Un array de NumPy con las series de tiempo,
                            donde las columnas son las variables y las filas son
                            las observaciones.
        max_rezagos (int): El número máximo de rezagos a evaluar.
        model (str): var, ar
    Returns:
        pandas.DataFrame: Una tabla con los valores de AIC, BIC y HQIC para
                          cada orden de rezago p.
    """


    resultados = []
    if verbose:
        print("Evaluando órdenes de rezago del 1 al {}...".format(max_rezagos))
    
    pbar = range(1, max_rezagos + 1)
    for p in pbar:
        if model_type == 'var':
            model = VAR(endog=datos)
            results = model.fit(maxlags=p)
            aic = results.aic
            bic = results.bic
            hqic = results.hqic
            
        elif model_type == 'ar':
            # Crear input si es necesario
            if inp_fn is not None:
                X, y = inp_fn(datos, break_point=break_point, p=p)
            else:
                continue

            model = sm.OLS(y, X)
            results = model.fit()
            logL = results.llf
            T = results.nobs
            k = results.df_model # Número de parámetros estimados
            
            aic = -2 * logL + 2 * k
            bic = -2 * logL + k * np.log(T)
            hqic = -2 * logL + 2 * k * np.log(np.log(T))

        else:
            raise ValueError("El modelo debe ser 'var' o 'ar'.")
        
        resultados.append({'p': p, 'AIC': aic, 'BIC': bic, 'HQIC': hqic})

    # Convertir la lista de resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados).set_index('p')
    if verbose:
        print("Evaluación completada.")
    return df_resultados

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



def find_best_p(serie, break_point, input_fn=None, p_max=12):
    """
    Encuentra el número óptimo de rezagos 'p' para un modelo específico.

    Args:
        serie (np.array): La serie de tiempo.
        break_point (int): El punto de quiebre a evaluar.
        input_fn (function): La función que genera X e y (create_model_a, b, o c).
        p_max (int): El número máximo de rezagos a probar.

    Returns:
        int: El número óptimo de rezagos 'p' según el criterio BIC.
    """
    metrics = []

    for p in range(p_max + 1):  # Probamos desde p=0 hasta p_max
        try:
            # 1. Construir la regresión COMPLETA para este 'p'
            if input_fn is not None:
                X, y = input_fn(serie=serie, break_point=break_point, p=p)

            if X.shape[0] < X.shape[1]: # No hay suficientes datos
                continue
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue # Salta esta iteración
            

            model = sm.OLS(y, X)
            results = model.fit()

            # bic = model.bic
            # aic = model.aic
            # hqc = model.hqic
            tstat = results.tvalues[3]

            # ols_model = OLS()
            # ols_model.fit(X, y)
            
            # T = X.shape[0]  # Nro de observaciones efectivas
            # k = X.shape[1]  # Nro de regresores (incluye dummies, lags, etc.)
            
            # residuals = y - ols_model.predict(X)
            # ssr = np.sum(residuals**2)
            # sigma2_hat = ssr / T

            # ols_model.beta

            # bic = np.log(sigma2_hat) + (k * np.log(T)) / T
            # aic = np.log(sigma2_hat) + (2 * k) / T
            # hqc = np.log(sigma2_hat) + (2 * k * np.log(np.log(T))) / T

            metrics.append({'p': p, 't':tstat})
        
        except Exception as e:
            print(e)
            continue

    if not metrics:
        return None

    results_df = pd.DataFrame(metrics)
    return results_df
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Valores críticos de KPSS
Valores_criticos = {
    "c": {  # constante
        0.10: 0.347,
        0.05: 0.463,
        0.025: 0.574,
        0.01: 0.739
    },
    "ct": {  # tendencia
        0.10: 0.119,
        0.05: 0.146,
        0.025: 0.176,
        0.01: 0.216
    }
}

def test_kpss(series, tipo="c", alpha=0.05):
    """
    Realiza el test de KPSS y devuelve el estadístico, p-value y una conclusión.
    """
    # 0. Preparación de datos
    y = pd.Series(series).dropna()
    T = len(y)

    results_models = []
    # 1. Regresión para especificación constante y tendencia
    for tipo in ['c', 'ct']:
        if tipo == "c":
            X = np.ones((T, 1))
        elif tipo == "ct":
            X = sm.add_constant(np.arange(1, T + 1))
        else:
            raise ValueError("El tipo debe ser 'c' o 'ct'.")
        
        modelo = sm.OLS(y, X).fit()

        # 2. Residuos
        u = modelo.resid

        # 3. Suma acumulada de los residuos 
        S = np.cumsum(u)

        # 4. Estimador HAC (Newey-West)
        lags = int(12 * (T / 100)**(1 / 4))
        # acovf devuelve una tupla, el primer elemento es el array de autocovarianzas
        s2 = sm.tsa.stattools.acovf(u, fft=False, nlag=lags)
        w = 1 - np.arange(1, lags + 1) / (lags + 1)
        sigma2 = s2[0] + 2 * np.sum(w * s2[1:])
        
        # Manejar el caso de varianza cero para evitar un error
        if sigma2 <= 0:
            stat = np.inf
        else:
            # 5. Estadístico KPSS
            stat = np.sum(S ** 2) / (T ** 2 * sigma2)

        # 6. Estimar p-value por interpolación
        # El test de KPSS es de cola derecha, por lo que los valores de `x`
        # (los críticos) deben estar en orden creciente.
        critical_values = np.array(list(Valores_criticos[tipo].values()))
        p_levels = np.array(list(Valores_criticos[tipo].keys()))
        
        # np.interp requiere que `xp` esté en orden ascendente.
        # Los valores críticos están en orden descendente, así que los invertimos.
        sorted_critical_values = np.flip(critical_values)
        sorted_p_levels = np.flip(p_levels)
        
        # Realizamos la interpolación
        p_value = np.interp(stat, sorted_critical_values, sorted_p_levels, left=0.01, right=0.10)

        # 7. Conclusión
        conclusion = "Estacionaria" if p_value < alpha else "No estacionaria"

        # Retornar los resultados en un DataFrame
        result_df = pd.DataFrame({
            'model': [tipo],
            'statistic': [stat],
            'p-value': [p_value],
            'conclusion': [conclusion]
        })
        results_models.append(result_df)
    return pd.concat(results_models, axis=0)
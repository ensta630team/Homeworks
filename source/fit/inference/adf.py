import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_adf(serie: np.ndarray, max_k: int = 10, model_type: str = 'c', alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza el Test de Dickey-Fuller Aumentado (ADF).

    Encuentra el número óptimo de rezagos (k) usando el test de Ljung-Box
    sobre los residuos, estima el modelo final y calcula el estadístico del test.

    Args:
        serie (np.ndarray): La serie de tiempo a analizar.
        max_k (int, optional): Máximo número de rezagos a considerar. Default es 10.
        model_type (str, optional): Tipo de regresión a usar. 
                                    'nc': sin constante.
                                    'c': con constante.
                                    'ct': con constante y tendencia.
                                    Default es 'c'.
        alpha (float, optional): Nivel de significancia para el test de ruido blanco.
                                 Default es 0.05.

    Returns:
        pd.DataFrame: Un DataFrame con las métricas del test.
    """

    # --- Función Auxiliar Anidada ---
    # Esta función solo es visible y utilizada por test_adf.
    def encontrar_k_optimo(series, max_k, alpha, estimar_constante, estimar_tendencia):
        """Encuentra el k mínimo para que los residuos sean ruido blanco."""
        yt = pd.Series(series, name='y')
        delta_yt = yt.diff()
        for k in range(max_k + 1):
            y_lag_1 = yt.shift(1).rename('y_lag_1')
            variables_a_unir = [yt.to_frame(), y_lag_1.to_frame()]
            
            if k > 0:
                lagged_deltas = pd.DataFrame()
                for i in range(1, k + 1):
                    lagged_deltas[f'delta_y_lag_{i}'] = delta_yt.shift(i)
                variables_a_unir.append(lagged_deltas)
            
            if estimar_tendencia:
                trend = pd.Series(range(len(yt)), index=yt.index, name='trend')
                variables_a_unir.append(trend.to_frame())

            full_df = pd.concat(variables_a_unir, axis=1).dropna()
            Y = full_df['y']
            X = full_df.drop('y', axis=1)

            if estimar_constante:
                X.insert(0, 'const', 1)

            beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
            residuals = Y - X @ beta
            
            lags_for_test = min(10, len(residuals) // 5)
            if lags_for_test < 1: continue
                
            p_value = acorr_ljungbox(residuals, lags=[lags_for_test], return_df=True)['lb_pvalue'].iloc[0]
            if p_value > alpha:
                return k
        return None
    
    # Determinar la especificación del modelo
    estimar_constante = model_type in ['c', 'ct']
    estimar_tendencia = model_type == 'ct'

    # Encontrar el k óptimo
    k_optimo = encontrar_k_optimo(serie, max_k, alpha, estimar_constante, estimar_tendencia)
    
    if k_optimo is None:
        # Si no se encuentra k, retornar un DataFrame con un error
        resultados = [{'Métrica': 'Error', 'Valor': 'No se encontró un k óptimo.'}]
        return pd.DataFrame(resultados)

    # Estimar el modelo final con el k óptimo
    yt = pd.Series(serie, name='y')
    delta_yt = yt.diff()
    y_lag_1 = yt.shift(1).rename('y_lag_1')
    
    variables_a_unir = [yt.to_frame(), y_lag_1.to_frame()]
    if k_optimo > 0:
        lagged_deltas = pd.DataFrame()
        for i in range(1, k_optimo + 1):
            lagged_deltas[f'delta_y_lag_{i}'] = delta_yt.shift(i)
        variables_a_unir.append(lagged_deltas)
    if estimar_tendencia:
        trend = pd.Series(range(len(yt)), index=yt.index, name='trend')
        variables_a_unir.append(trend.to_frame())
        
    full_df = pd.concat(variables_a_unir, axis=1).dropna()
    Y = full_df['y']
    X = full_df.drop('y', axis=1)
    if estimar_constante:
        X.insert(0, 'const', 1)
    
    beta_estimado = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    residuals = Y - X @ beta_estimado
    
    # Calcular el estadístico y otras métricas
    n_obs, n_params = X.shape
    res_var = np.sum(residuals**2) / (n_obs - n_params)
    cov_matrix = res_var * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_matrix))
    
    idx_phi = X.columns.get_loc('y_lag_1')
    phi_estimado = beta_estimado[idx_phi]
    se_phi = se[idx_phi]
    
    estadistico_adf = (phi_estimado - 1) / se_phi
    
    critical_values_dict = {
        'nc': {'1%': -2.58, '5%': -1.95, '10%': -1.62},
        'c':  {'1%': -3.43, '5%': -2.86, '10%': -2.57},
        'ct': {'1%': -3.96, '5%': -3.41, '10%': -3.12}
    }
    valores_criticos = critical_values_dict[model_type]
    
    # Formatear la salida según el estándar requerido
    resultados = [
        {'Métrica': 'Estadístico', 'Valor': estadistico_adf},
        {'Métrica': 'Rezagos Óptimos (k)', 'Valor': k_optimo},
        {'Métrica': 'Valor Crítico (1%)', 'Valor': valores_criticos['1%']},
        {'Métrica': 'Valor Crítico (5%)', 'Valor': valores_criticos['5%']},
        {'Métrica': 'Valor Crítico (10%)', 'Valor': valores_criticos['10%']},
        {'Métrica': 'Nota', 'Valor': 'El p-valor no se calcula en esta implementación manual.'}
    ]

    return pd.DataFrame(resultados)

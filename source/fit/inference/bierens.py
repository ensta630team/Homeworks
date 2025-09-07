import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_bierens(serie: np.ndarray, max_k: int = 10, max_m: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza el Test de Bierens (1997) de forma manual.

    Prueba la hipótesis nula de raíz unitaria contra la alternativa de que la
    serie es estacionaria alrededor de una tendencia no lineal determinista.
    Encuentra los órdenes óptimos (m, k) usando el test de Ljung-Box sobre
    los residuos, estima el modelo final y calcula los estadísticos del test.

    Args:
        serie (np.ndarray): La serie de tiempo a analizar.
        max_k (int, optional): Máximo número de rezagos ADF a considerar. Default es 10.
        max_m (int, optional): Máximo orden de los polinomios de Chebyshov a usar. Default es 5.
        alpha (float, optional): Nivel de significancia para el test de ruido blanco y
                                 para seleccionar el valor crítico. Default es 0.05.

    Returns:
        pd.DataFrame: Un DataFrame con las métricas del test.
    """

    # --- Funciones Auxiliares Anidadas ---

    def polinomios_chebyshev(T, p):
        """Genera la matriz de regresores de tendencia Pol."""
        t_vals = np.arange(1, T + 1)
        k_vals = np.arange(1, p + 1)
        i_grid, k_grid = np.meshgrid(t_vals, k_vals, indexing='ij')
        P = np.sqrt(2) * np.cos(k_grid * np.pi * (i_grid - 0.5) / T)

        tn = t_vals / T
        x0 = np.ones(T)
        num_polinomios_a_limpiar = p // 2
        e = np.zeros((T, num_polinomios_a_limpiar))

        for k in range(1, num_polinomios_a_limpiar + 1):
            y = P[:, 2*k - 2]
            if k == 1:
                X_clean = np.column_stack([x0, tn])
            else:
                indices = np.arange(1, 2*(k-1), 2)
                prev_P_odd = P[:, indices]
                X_clean = np.column_stack([x0, prev_P_odd, tn])

            beta = np.linalg.inv(X_clean.T @ X_clean) @ (X_clean.T @ y)
            residuals = y - X_clean @ beta
            rmse = np.sqrt(np.sum(residuals**2) / T)
            e[:, k-1] = residuals / rmse

        POL = np.zeros((T, p))
        for i in range(2, p + 1, 2):
            k = i // 2
            POL[:, i-1] = e[:, k-1]
        if p > 2:
            for i in range(2, p - 1, 2):
                k_impar = i + 1
                POL[:, k_impar-1] = P[:, i-1]
                
        mean_t = (T + 1) / 2
        std_t = np.sqrt((T**2 - 1) / 12)
        POL[:, 0] = (t_vals - mean_t) / std_t
        
        return pd.DataFrame(POL, columns=[f'POL_{i+1}' for i in range(p)])

    def encontrar_km_optimos(series, max_k, max_m, alpha):
        """Encuentra la combinación (m, k) más parsimoniosa."""
        yt = pd.Series(series, name='y')
        delta_yt = yt.diff().rename('delta_y')
        T = len(yt)
        for m in range(max_m + 1):
            matriz_Pol = None
            if m > 0:
                p_orden = m * 2
                matriz_Pol = polinomios_chebyshev(T, p_orden)
            for k in range(max_k + 1):
                y_lag_1 = yt.shift(1).rename('y_lag_1')
                variables_a_unir = [delta_yt.to_frame(), y_lag_1.to_frame()]
                if k > 0:
                    lagged_deltas = pd.DataFrame()
                    for i in range(1, k + 1):
                        lagged_deltas[f'delta_y_lag_{i}'] = delta_yt.shift(i)
                    variables_a_unir.append(lagged_deltas)
                if m > 0:
                    terminos_tendencia = matriz_Pol.iloc[:, :m+1]
                    variables_a_unir.append(terminos_tendencia)

                full_df = pd.concat(variables_a_unir, axis=1).dropna()
                Y = full_df['delta_y']
                X = full_df.drop('delta_y', axis=1)
                X.insert(0, 'const', 1)
                beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
                residuals = Y - X @ beta
                                
                lags_for_test = min(10, len(residuals) // 5)
                if lags_for_test < 1: continue
                p_value = acorr_ljungbox(residuals, lags=[lags_for_test], return_df=True)['lb_pvalue'].iloc[0]
                if p_value > alpha:
                    return m, k
        return None, None

    # --- Lógica Principal del Test de Bierens ---

    # 1. Encontrar los órdenes óptimos
    m_optimo, k_optimo = encontrar_km_optimos(serie, max_k, max_m, alpha)
    
    if m_optimo is None:
        resultados = [{'Métrica': 'Error', 'Valor': 'No se encontró una combinación (m, k) óptima.'}]
        return pd.DataFrame(resultados)

    # 2. Estimar el modelo final con los (m, k) óptimos
    yt = pd.Series(serie, name='y')
    delta_yt = yt.diff().rename('delta_y')
    T = len(yt)
    matriz_Pol = polinomios_chebyshev(T, m_optimo * 2) if m_optimo > 0 else None
        
    y_lag_1 = yt.shift(1).rename('y_lag_1')
    variables_a_unir = [delta_yt.to_frame(), y_lag_1.to_frame()]

    if k_optimo > 0:
        lagged_deltas = pd.DataFrame()
        for i in range(1, k_optimo + 1):
            lagged_deltas[f'delta_y_lag_{i}'] = delta_yt.shift(i)
        variables_a_unir.append(lagged_deltas)
    if m_optimo > 0:
        terminos_tendencia = matriz_Pol.iloc[:, :m_optimo+1]
        variables_a_unir.append(terminos_tendencia)

    full_df = pd.concat(variables_a_unir, axis=1).dropna()
    Y = full_df['delta_y']
    X = full_df.drop('delta_y', axis=1)
    X.insert(0, 'const', 1)
    n_obs = len(Y)
    
    beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    
    # 3. Calcular ambos estadísticos
    residuals = Y - X @ beta
    res_var = np.sum(residuals**2) / (n_obs - len(beta))
    cov_matrix = res_var * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_matrix))
    idx_rho = X.columns.get_loc('y_lag_1')
    rho_estimado = beta[idx_rho]
    
    t_stat = rho_estimado / se[idx_rho]
    
    suma_coef_lags = 0
    if k_optimo > 0:
        nombres_lags = [f'delta_y_lag_{i}' for i in range(1, k_optimo + 1)]
        indices_lags = [X.columns.get_loc(nombre) for nombre in nombres_lags]
        suma_coef_lags = np.sum(beta[indices_lags])
    
    rho_stat = (n_obs * rho_estimado) / (1 - suma_coef_lags) if (1 - suma_coef_lags) != 0 else np.nan
    
    critical_values_dict = {
        0: {'1%': -3.43, '5%': -2.86, '10%': -2.57}, 1: {'1%': -3.76, '5%': -3.20, '10%': -2.90},
        2: {'1%': -4.01, '5%': -3.44, '10%': -3.13}, 3: {'1%': -4.22, '5%': -3.64, '10%': -3.32},
        4: {'1%': -4.41, '5%': -3.81, '10%': -3.50}, 5: {'1%': -4.58, '5%': -3.97, '10%': -3.66}
    }
    valores_criticos = critical_values_dict.get(m_optimo, {})

    # 4. Formatear la salida según el estándar requerido
    resultados = [
        {'Métrica': 'Estadístico (Sesgo Normalizado)', 'Valor': rho_stat},
        {'Métrica': 'Estadístico (t-stat)', 'Valor': t_stat},
        {'Métrica': 'Rezagos Óptimos (k)', 'Valor': k_optimo},
        {'Métrica': 'Polinomios Óptimos (m)', 'Valor': m_optimo},
        {'Métrica': 'Valor Crítico (1%)', 'Valor': valores_criticos.get('1%')},
        {'Métrica': 'Valor Crítico (5%)', 'Valor': valores_criticos.get('5%')},
        {'Métrica': 'Valor Crítico (10%)', 'Valor': valores_criticos.get('10%')},
        {'Métrica': 'Nota', 'Valor': 'El p-valor no se calcula en esta implementación manual.'}
    ]

    return pd.DataFrame(resultados)
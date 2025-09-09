import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_bierens(serie: np.ndarray, m: int = None, k: int = None, max_k: int = 10, max_m: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza el Test de Bierens (1997) para un m y k específicos, o los busca.

    Prueba la hipótesis nula de raíz unitaria contra la alternativa de que la
    serie es estacionaria alrededor de una tendencia no lineal. Reporta el
    estadístico de sesgo normalizado (Z_rho) y aproxima su p-valor.

    Args:
        serie (np.ndarray): La serie de tiempo a analizar.
        m (int, optional): Orden de los polinomios a usar. Si es None, se busca.
        k (int, optional): Número de rezagos ADF. Si es None, se busca.
        max_k (int, optional): Máximo número de rezagos a buscar. Default es 10.
        max_m (int, optional): Máximo orden de polinomios a buscar. Default es 10.
        alpha (float, optional): Nivel de significancia. Default es 0.05.

    Returns:
        pd.DataFrame: Un DataFrame con el resultado del test.
    """

    # --- Tabla de Valores Críticos para Z_rho (Bierens, 1997, Tabla 1, p. 55) ---
    BIERENS_CV_RHO = {
        1: {'1%': -29.7, '5%': -22.0, '10%': -18.3},
        2: {'1%': -37.0, '5%': -27.2, '10%': -23.0},
        3: {'1%': -46.2, '5%': -35.6, '10%': -30.9},
        4: {'1%': -52-2, '5%': -41.6, '10%': -36.6},
        5: {'1%': -61.1, '5%': -48.7, '10%': -43.4},
        6: {'1%': -66.9, '5%': -54.7, '10%': -49.1},
        7: {'1%': -74.8, '5%': -61.8, '10%': -55.8},
        8: {'1%': -80.6, '5%': -67.9, '10%': -61.7},
        9: {'1%': -88.9, '5%': -74.4, '10%': -67.7},
       10: {'1%': -94.2, '5%': -80.3, '10%': -73.7}
    }
    ADF_CV_C = {'1%': -3.43, '5%': -2.86, '10%': -2.57} # Para el caso m=0 (Z_t)

    def polinomios_chebyshev(T, p):
        if p == 0: return None
        t_vals = np.arange(1, T + 1)
        k_vals = np.arange(1, p + 1)
        i_grid, k_grid = np.meshgrid(t_vals, k_vals, indexing='ij')
        P = np.sqrt(2) * np.cos(k_grid * np.pi * (i_grid - 0.5) / T)
        tn = t_vals / T; x0 = np.ones(T)
        num_limpiar = p // 2
        e = np.zeros((T, num_limpiar))
        for i_k in range(1, num_limpiar + 1):
            y = P[:, 2*i_k - 2]
            if i_k == 1: X_c = np.column_stack([x0, tn])
            else:
                indices = np.arange(1, 2*(i_k-1), 2)
                X_c = np.column_stack([x0, P[:, indices], tn])
            beta = np.linalg.inv(X_c.T @ X_c) @ (X_c.T @ y)
            res = y - X_c @ beta
            e[:, i_k-1] = res / np.sqrt(np.sum(res**2) / T)
        POL = np.zeros((T, p))
        for i in range(2, p + 1, 2): POL[:, i-1] = e[:, i//2 - 1]
        if p > 2:
            for i in range(2, p - 1, 2): POL[:, i] = P[:, i-1]
        POL[:, 0] = (t_vals - (T + 1) / 2) / np.sqrt((T**2 - 1) / 12)
        return pd.DataFrame(POL, columns=[f'POL_{i+1}' for i in range(p)])

    def _encontrar_km_optimos(series, max_k, max_m, alpha):
        yt = pd.Series(series, name='y'); delta_yt = yt.diff().rename('delta_y'); T = len(yt)
        for i_m in range(max_m + 1):
            matriz_Pol = polinomios_chebyshev(T, i_m * 2) if i_m > 0 else None
            for i_k in range(max_k + 1):
                y_lag_1 = yt.shift(1).rename('y_lag_1')
                vars_unir = [delta_yt.to_frame(), y_lag_1.to_frame()]
                if i_k > 0:
                    lags = pd.DataFrame({f'd_y_lag_{i}': delta_yt.shift(i) for i in range(1, i_k + 1)})
                    vars_unir.append(lags)
                if i_m > 0: vars_unir.append(matriz_Pol.iloc[:, :i_m+1])
                df = pd.concat(vars_unir, axis=1).dropna()
                Y = df['delta_y']; X = df.drop('delta_y', axis=1); X.insert(0, 'const', 1)
                beta = np.linalg.inv(X.T @ X) @ (X.T @ Y); res = Y - X @ beta
                p_val = acorr_ljungbox(res, lags=[min(10, len(res)//5)], return_df=True)['lb_pvalue'].iloc[0]
                if p_val > alpha: return i_m, i_k
        return None, None

    def _interpolar_p_valor(estadistico, m):
        if m > 0:
            if m not in BIERENS_CV_RHO: return np.nan # No hay CVs para este m
            cv = BIERENS_CV_RHO[m]
        else: # m == 0, se usa la tabla ADF
            cv = ADF_CV_C

        cv1, cv5, cv10 = cv['1%'], cv['5%'], cv['10%']
        p_levels = np.array([0.01, 0.05, 0.10])
        critical_values = np.array([cv1, cv5, cv10])
        
        return np.interp(estadistico, critical_values, p_levels, right=1.0)

    # --- Lógica Principal del Test ---
    if m is None or k is None:
        m_optimo, k_optimo = _encontrar_km_optimos(serie, max_k, max_m, alpha)
    else:
        m_optimo, k_optimo = m, k

    nombre_modelo = f'Bierens(m={m_optimo}, k={k_optimo})'
    try:
        if m_optimo is None: raise ValueError("No se encontró (m,k) óptimo.")
        
        yt = pd.Series(serie, name='y'); delta_yt = yt.diff().rename('delta_y'); T = len(yt)
        matriz_Pol = polinomios_chebyshev(T, m_optimo * 2) if m_optimo > 0 else None
        y_lag_1 = yt.shift(1).rename('y_lag_1'); vars_unir = [delta_yt.to_frame(), y_lag_1.to_frame()]
        if k_optimo > 0:
            lags = pd.DataFrame({f'd_y_lag_{i}': delta_yt.shift(i) for i in range(1, k_optimo + 1)})
            vars_unir.append(lags)
        if m_optimo > 0: vars_unir.append(matriz_Pol.iloc[:, :m_optimo+1])
        full_df = pd.concat(vars_unir, axis=1).dropna(); n_obs = len(full_df)
        Y = full_df['delta_y']; X = full_df.drop('delta_y', axis=1); X.insert(0, 'const', 1)
        
        beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
        idx_rho = X.columns.get_loc('y_lag_1'); rho_estimado = beta[idx_rho]
        
        # El caso m=0 es especial: se reporta el t-stat (Z_t)
        if m_optimo == 0:
            res = Y - X @ beta; res_var = np.sum(res**2)/(n_obs-len(beta)); cov = res_var*np.linalg.inv(X.T @ X)
            stat_final = rho_estimado / np.sqrt(np.diag(cov))[idx_rho]
            nombre_modelo = f'Bierens(m=0, k={k_optimo})' # Es un ADF, no Bierens
        else: # Para m > 0, se reporta Z_rho
            suma_coef_lags = 0
            if k_optimo > 0:
                nombres_lags = [col for col in X.columns if 'd_y_lag' in col]
                indices_lags = [X.columns.get_loc(nombre) for nombre in nombres_lags]
                suma_coef_lags = np.sum(beta[indices_lags])
            stat_final = (n_obs * rho_estimado) / (1 - suma_coef_lags) if (1 - suma_coef_lags) != 0 else np.nan

        p_valor_aprox = _interpolar_p_valor(stat_final, m_optimo)
        resultado = {'model': nombre_modelo, 'statistic': stat_final, 'p-value': p_valor_aprox}

    except (np.linalg.LinAlgError, ValueError):
        resultado = {'model': nombre_modelo, 'statistic': None, 'p-value': None}

    return pd.DataFrame([resultado])
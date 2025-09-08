import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_adf(serie: np.ndarray, max_k: int = 10, model_type: str = 'c', alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza el Test de Dickey-Fuller Aumentado (ADF) para un tipo de modelo específico.

    Calcula los valores críticos ajustados por tamaño muestral y aproxima el p-valor
    usando una interpolación lineal robusta.

    Args:
        serie (np.ndarray): La serie de tiempo a analizar.
        max_k (int, optional): Máximo número de rezagos a considerar. Default es 10.
        model_type (str, optional): Tipo de regresión a usar ('nc', 'c', 'ct').
                                    Default es 'c'.
        alpha (float, optional): Nivel de significancia para el test de ruido blanco.
                                 Default es 0.05.

    Returns:
        pd.DataFrame: Un DataFrame con el resultado del test solicitado.
    """

    # --- Coeficientes de MacKinnon (1996) ---
    MACKINNON_COEFS = {
        'nc': {'1%': (-2.56574, -3.75, -15.0), '5%': (-1.94100, -1.13, -5.0), '10%': (-1.61682, -0.51, -2.8)},
        'c':  {'1%': (-3.43035, -17.9, -85.0), '5%': (-2.86154, -6.5, -33.0), '10%': (-2.56677, -2.8, -15.0)},
        'ct': {'1%': (-3.95877, -26.8, -130.0), '5%': (-3.41049, -12.5, -60.0), '10%': (-3.12705, -7.0, -36.0)}
    }

    # --- Funciones Auxiliares Anidadas ---
    def _encontrar_k_optimo(series, max_k, alpha, estimar_constante, estimar_tendencia):
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
            if estimar_constante: X.insert(0, 'const', 1)
            beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
            residuals = Y - X @ beta
            lags_for_test = min(10, len(residuals) // 5)
            if lags_for_test < 1: continue
            p_value = acorr_ljungbox(residuals, lags=[lags_for_test], return_df=True)['lb_pvalue'].iloc[0]
            if p_value > alpha: return k
        return None

    def _get_p_value(estadistico, T, model_type):
        """Calcula valores críticos precisos y el p-valor interpolado."""
        def get_cv(level):
            c_inf, c_1, c_2 = MACKINNON_COEFS[model_type][level]
            return c_inf + c_1 / T + c_2 / (T**2)
        
        cv_1, cv_5, cv_10 = get_cv('1%'), get_cv('5%'), get_cv('10%')
        critical_values_points = np.array([cv_1, cv_5, cv_10])
        p_levels_points = np.array([0.01, 0.05, 0.10])
        
        p_value = np.interp(estadistico, critical_values_points, p_levels_points, right=1.0)
        return p_value

    # --- Lógica Principal del Test ---
    nombre_modelo = f'ADF-{model_type}'
    try:
        estimar_constante = model_type in ['c', 'ct']
        estimar_tendencia = model_type == 'ct'
        k_optimo = _encontrar_k_optimo(serie, max_k, alpha, estimar_constante, estimar_tendencia)
        if k_optimo is None: raise ValueError("No se encontró k óptimo.")

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
        if estimar_constante: X.insert(0, 'const', 1)

        beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
        residuals = Y - X @ beta
        res_var = np.sum(residuals**2) / (len(Y) - X.shape[1])
        cov_matrix = res_var * np.linalg.inv(X.T @ X)
        idx_phi = X.columns.get_loc('y_lag_1')
        se_phi = np.sqrt(np.diag(cov_matrix))[idx_phi]
        phi_estimado = beta[idx_phi]
        estadistico_adf = (phi_estimado - 1) / se_phi
        
        T_efectivo = len(Y)
        p_valor_aprox = _get_p_value(estadistico_adf, T_efectivo, model_type)

        resultado = {'model': nombre_modelo, 'statistic': estadistico_adf, 'p-value': p_valor_aprox}

    except (np.linalg.LinAlgError, ValueError):
        resultado = {'model': nombre_modelo, 'statistic': None, 'p-value': None}

    return pd.DataFrame([resultado])
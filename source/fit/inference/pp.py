import numpy as np
import pandas as pd

def test_pp(A, q=4, regression='all', alpha=0.05):
    """
    Realiza el test de raíz unitaria de Phillips-Perron y retorna una
    tabla completa con resultados, decisión e interpretación.

    Hipótesis Nula (H₀): La serie tiene una raíz unitaria (no es estacionaria).
    Hipótesis Alternativa (H₁): La serie es estacionaria.

    Parámetros:
    -----------
    A : array-like
        La serie de tiempo a analizar.
    q : int, opcional
        El número de rezagos para el estimador de varianza de largo plazo.
    regression : {'case1', 'case2', 'case4', 'all'}, opcional
        El tipo de regresión a realizar.
    alpha : float, opcional
        Nivel de significancia para la decisión (e.g., 0.01, 0.05, 0.10).

    Retorna:
    --------
    pandas.DataFrame
        Una tabla ordenada con el valor del estadístico Z-t, valor crítico,
        p-valor aproximado, decisión del test e interpretación.
    """
    print("Test PP")
    # Coeficientes de MacKinnon (1996) para los valores críticos del Z-t
    MACKINNON_COEFS = {
        'case1': { # Sin constante
            '1%': [-2.5657, -3.76, -11.0], 
            '5%': [-1.9410, -1.13, -2.8], 
            '10%': [-1.6168, -0.36, -1.1]
        },
        'case2': { # Con constante
            '1%': [-3.4303, -17.86, -86.8], '5%': [-2.8615, -6.49, -27.2], '10%': [-2.5668, -3.52, -12.9]
        },
        'case4': { # Con constante y tendencia
            '1%': [-3.9588, -28.69, -187.9], '5%': [-3.4105, -12.58, -63.3], '10%': [-3.1271, -8.11, -39.0]
        }
    }
    
    # --- Funciones auxiliares ---
    def _calculate_pp_stats(residuals, T, q, rho, var_rho, s2):
        """Calcula los estadísticos Z-rho y Z-t."""
        def calculate_autocov(res, lag, n_obs):
            return np.dot(res[lag:], res[:-lag]) / n_obs if lag > 0 else np.dot(res, res) / n_obs

        autocov = np.array([calculate_autocov(residuals, j, T) for j in range(q + 1)])
        gamma0 = autocov[0]
        gamma_j = autocov[1:]
        weights = 1 - np.arange(1, q + 1) / (q + 1)
        lambda2 = gamma0 + 2 * np.sum(weights * gamma_j)

        if lambda2 < 1e-9: lambda2 = 1e-9
        
        se_rho = np.sqrt(var_rho)
        z_t = (((rho - 1) / se_rho) * np.sqrt(gamma0 / lambda2) -
               (T * se_rho / np.sqrt(lambda2 * s2)) * (lambda2 - gamma0) * 0.5)
        return z_t

    def _get_decision_details(z_t, case, T, alpha):
        """Calcula valores críticos, p-valor interpolado y genera la decisión."""
        def get_cv(level):
            c_inf, c_1, c_2 = MACKINNON_COEFS[case][level]
            return c_inf + c_1 / T + c_2 / (T**2)

        cv_1, cv_5, cv_10 = get_cv('1%'), get_cv('5%'), get_cv('10%')
        
        critical_values = np.array([cv_1, cv_5, cv_10])
        p_levels = np.array([0.01, 0.05, 0.10])
        p_value = np.interp(z_t, critical_values, p_levels, right=1.0)
            
        critical_value_map = {0.01: cv_1, 0.05: cv_5, 0.10: cv_10}
        critical_value = critical_value_map.get(alpha, get_cv(f'{int(alpha*100)}%'))
            
        if p_value < alpha:
            decision = f'Rechazar H₀ (α={alpha})'
            interpretation = 'La serie es estacionaria'
        else:
            decision = f'No Rechazar H₀ (α={alpha})'
            interpretation = 'La serie tiene una raíz unitaria'
            
        return critical_value, p_value, decision, interpretation

    # --- Preparación de datos y ejecución ---
    A = np.asarray(A, dtype=float).flatten()
    Y, T, Rezago = A[1:], len(A[1:]), A[:-1]
    
    resultados_finales = []
    
    # Diccionario para mapear regresiones
    cases = {
        'case1': ('pp - Sin constante ni tendencia', Rezago.reshape(-1, 1), 0),
        'case2': ('pp - Con constante', np.column_stack([np.ones(T), Rezago]), 1),
        'case4': ('pp - Con constante y tendencia', np.column_stack([np.ones(T), np.arange(1, T + 1), Rezago]), 2)
    }

    regressions_to_run = cases.keys() if regression == 'all' else [regression]

    for case_key in regressions_to_run:
        nombre, X, rho_idx = cases[case_key]
        k = X.shape[1]
        
        try:
            Beta = np.linalg.solve(X.T @ X, X.T @ Y)
            rho = Beta[rho_idx]
            Resid = Y - X @ Beta
            s2 = np.dot(Resid, Resid) / (T - k)
            VarMCO = s2 * np.linalg.inv(X.T @ X)
            var_rho = VarMCO[rho_idx, rho_idx]
            
            z_t = _calculate_pp_stats(Resid, T, q, rho, var_rho, s2)
            cv, pval, dec, interp = _get_decision_details(z_t, case_key, T, alpha)
            
            resultados_finales.append({
                'model': nombre,
                'statistic': z_t,
                'p-value': pval,
            })

        except np.linalg.LinAlgError:
            # En caso de multicolinealidad o matriz singular
            resultados_finales.append({
                'model': nombre,
                'statistic': None,
                'p-value': None,
            })

    return pd.DataFrame(resultados_finales)

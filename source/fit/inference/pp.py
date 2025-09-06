import numpy as np
import pandas as pd
from scipy import stats

def test_pp(A, q=4, regression='all'):
    """
    Phillips-Perron test for unit roots
    
    Parameters:
    -----------
    A : np.ndarray
        Time series data
    q : int, optional
        Number of lags for autocovariances (default=4)
    regression : str, optional
        Regression type: 'case1', 'case2', 'case4', or 'all' (default='all')
        
    Returns:
    --------
    pd.DataFrame with columns 'Metrica' and 'Valor'
    """
    
    A = np.array(A)
    Y = A[1:]
    T = len(Y)
    Cte = np.ones(T)
    Trend = np.arange(1, T+1)
    Rezago = A[:-1]
    
    # Design matrices
    X1 = Rezago.reshape(-1, 1)
    X2 = np.column_stack([Cte, Rezago])
    X4 = np.column_stack([Cte, Rezago, Trend])
    
    # OLS estimation
    Beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ Y if regression in ['case1', 'all'] else None
    Beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ Y if regression in ['case2', 'all'] else None
    Beta4 = np.linalg.inv(X4.T @ X4) @ X4.T @ Y if regression in ['case4', 'all'] else None
    
    metricas = []
    valores = []
    
    def add_result(metric, value):
        metricas.append(metric)
        valores.append(value)
    
    # Critical values
    critical_values = {
        'case1': {'1%': -11.8, '5%': -7.3, '10%': -5.3},
        'case2': {'1%': -13.8, '5%': -8.1, '10%': -5.7},
        'case4': {'1%': -20.7, '5%': -14.1, '10%': -11.3}
    }
    
    # Asymptotic parameters for p-value approximation
    asymptotic_params = {
        'case1': {'mean': -2.5, 'std': 3.0},
        'case2': {'mean': -3.0, 'std': 3.5},
        'case4': {'mean': -4.0, 'std': 4.0}
    }
    
    def calculate_p_value(statistic, case):
        params = asymptotic_params[case]
        p_value = stats.norm.cdf(statistic, loc=params['mean'], scale=params['std'])
        return min(p_value, 1 - p_value) * 2
    
    # Case 1: No constant, no trend
    if regression in ['case1', 'all'] and Beta1 is not None:
        rho1 = Beta1[0]
        Resid1 = Y - X1 @ Beta1
        k1 = len(Beta1)
        s2_1 = (Resid1.T @ Resid1) / (T - k1)
        VarMCO1 = s2_1 * np.linalg.inv(X1.T @ X1)
        sigmarho1 = np.sqrt(VarMCO1[0, 0])
        
        AutoCov1 = np.zeros(q + 1)
        for j in range(q + 1):
            if j == 0:
                AutoCov1[j] = np.mean(Resid1 * Resid1)
            else:
                AutoCov1[j] = np.mean(Resid1[j:] * Resid1[:-j])
        
        lambda2_1 = AutoCov1[0] + 2 * sum([(1 - (k / (q + 1))) * AutoCov1[k] for k in range(1, q + 1)])
        Zrho1 = T * (rho1 - 1) - 0.5 * ((T**2) * (sigmarho1**2) / s2_1) * (lambda2_1 - AutoCov1[0])
        
        p_value_zrho1 = calculate_p_value(Zrho1, 'case1')
        
        add_result('Estadistico Zρ Case1', Zrho1)
        add_result('Valor P Zρ Case1', p_value_zrho1)
        add_result('ρ estimado Case1', rho1)
        
        for level, value in critical_values['case1'].items():
            add_result(f'Valor Critico {level} Case1', value)
    
    # Case 2: With constant
    if regression in ['case2', 'all'] and Beta2 is not None:
        rho2 = Beta2[1]
        Resid2 = Y - X2 @ Beta2
        k2 = len(Beta2)
        s2_2 = (Resid2.T @ Resid2) / (T - k2)
        VarMCO2 = s2_2 * np.linalg.inv(X2.T @ X2)
        sigmarho2 = np.sqrt(VarMCO2[1, 1])
        
        AutoCov2 = np.zeros(q + 1)
        for j in range(q + 1):
            if j == 0:
                AutoCov2[j] = np.mean(Resid2 * Resid2)
            else:
                AutoCov2[j] = np.mean(Resid2[j:] * Resid2[:-j])
        
        lambda2_2 = AutoCov2[0] + 2 * sum([(1 - (k / (q + 1))) * AutoCov2[k] for k in range(1, q + 1)])
        Zrho2 = T * (rho2 - 1) - 0.5 * ((T**2) * (sigmarho2**2) / s2_2) * (lambda2_2 - AutoCov2[0])
        
        p_value_zrho2 = calculate_p_value(Zrho2, 'case2')
        
        add_result('Estadistico Zρ Case2', Zrho2)
        add_result('Valor P Zρ Case2', p_value_zrho2)
        add_result('ρ estimado Case2', rho2)
        
        for level, value in critical_values['case2'].items():
            add_result(f'Valor Critico {level} Case2', value)
    
    # Case 4: With constant and trend
    if regression in ['case4', 'all'] and Beta4 is not None:
        rho4 = Beta4[1]
        Resid4 = Y - X4 @ Beta4
        k4 = len(Beta4)
        s2_4 = (Resid4.T @ Resid4) / (T - k4)
        VarMCO4 = s2_4 * np.linalg.inv(X4.T @ X4)
        sigmarho4 = np.sqrt(VarMCO4[1, 1])
        
        AutoCov4 = np.zeros(q + 1)
        for j in range(q + 1):
            if j == 0:
                AutoCov4[j] = np.mean(Resid4 * Resid4)
            else:
                AutoCov4[j] = np.mean(Resid4[j:] * Resid4[:-j])
        
        lambda2_4 = AutoCov4[0] + 2 * sum([(1 - (k / (q + 1))) * AutoCov4[k] for k in range(1, q + 1)])
        Zrho4 = T * (rho4 - 1) - 0.5 * ((T**2) * (sigmarho4**2) / s2_4) * (lambda2_4 - AutoCov4[0])
        
        term1 = ((rho4 - 1) / sigmarho4) * np.sqrt(AutoCov4[0] / lambda2_4)
        term2 = 0.5 * ((lambda2_4 - AutoCov4[0]) / np.sqrt(lambda2_4)) * (T * sigmarho4 / np.sqrt(s2_4))
        Zt4 = term1 - term2
        
        p_value_zrho4 = calculate_p_value(Zrho4, 'case4')
        p_value_zt4 = calculate_p_value(Zt4, 'case4')
        
        add_result('Estadistico Zρ Case4', Zrho4)
        add_result('Valor P Zρ Case4', p_value_zrho4)
        add_result('Estadistico Zt Case4', Zt4)
        add_result('Valor P Zt Case4', p_value_zt4)
        add_result('ρ estimado Case4', rho4)
        
        for level, value in critical_values['case4'].items():
            add_result(f'Valor Critico {level} Case4', value)
    
    add_result('Observaciones', T)
    add_result('Rezagos (q)', q)
    add_result('Tipo de regresion', regression)
    
    return pd.DataFrame({'Metrica': metricas, 'Valor': valores})

# Ejemplo de uso
if __name__ == "__main__":
    np.random.seed(123)
    n = 100
    y = np.cumsum(np.random.normal(0, 1, n))
    resultados = test_pp(y, q=4, regression='case2')
    print("=== Test de Phillips-Perron ===")
    print(resultados.to_string(index=False))
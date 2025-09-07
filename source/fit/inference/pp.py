import numpy as np
import pandas as pd

def test_pp(A, q=4, regression='all'):
    """
    Test de Phillips-Perron
    """
    
    A = np.asarray(A, dtype=float)
    Y = A[1:]
    T = len(Y)
    Cte = np.ones(T)
    Trend = np.arange(1, T + 1)
    Rezago = A[:-1]
    
    # Se definen las matrices de regresores
    X1 = Rezago.reshape(-1, 1)
    X2 = np.column_stack([Cte, Rezago])
    X4 = np.column_stack([Cte, Rezago, Trend])
    
    # Estimación MCO
    Beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ Y if regression in ['case1', 'all'] else None
    Beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ Y if regression in ['case2', 'all'] else None
    Beta4 = np.linalg.inv(X4.T @ X4) @ X4.T @ Y if regression in ['case4', 'all'] else None
    
    # Contenedores para los resultados
    resultados = []
    
    def calculate_autocov(residuals, lag, T):
        """Calculo de la autocovarianza sum(u_t * u_{t-j}) / T."""
        if lag == 0:
            return np.dot(residuals, residuals) / T
        return np.dot(residuals[lag:], residuals[:-lag]) / T

    # --- Caso 1: Sin constante ni tendencia ---
    if regression in ['case1', 'all'] and Beta1 is not None:
        rho1 = Beta1[0]
        Resid1 = Y - X1 @ Beta1
        k1 = X1.shape[1]
        s2_1 = (Resid1.T @ Resid1) / (T - k1)
        VarMCO1 = s2_1 * np.linalg.inv(X1.T @ X1)
        sigmarho1 = np.sqrt(VarMCO1[0, 0])
        
        AutoCov1 = np.array([calculate_autocov(Resid1, j, T) for j in range(q + 1)])
        
        # --- Cálculo de Lambda^2  ---
        aux1 = np.zeros(q)
        for j in range(1, q + 1):
            aux1[j-1] = (1 - (j / (q + 1))) * AutoCov1[j-1] 
        
        lambda2_1 = np.zeros(q)
        for j in range(1, q + 1):
            lambda2_1[j-1] = AutoCov1[0] + 2 * np.sum(aux1[0:j])
        # --------------------------------------------------------------------

        Zrho1 = T * (rho1 - 1) - 0.5 * ((T**2 * VarMCO1[0, 0]) / s2_1) * (lambda2_1 - AutoCov1[0])
        
        for i in range(q):
            resultados.append({'Metrica': f'Zrho Case1 (q={i+1})', 'Valor': Zrho1[i]})

    # --- Caso 2: Con constante ---
    if regression in ['case2', 'all'] and Beta2 is not None:
        rho2 = Beta2[1]
        Resid2 = Y - X2 @ Beta2
        k2 = X2.shape[1]
        s2_2 = (Resid2.T @ Resid2) / (T - k2)
        VarMCO2 = s2_2 * np.linalg.inv(X2.T @ X2)
        sigmarho2 = np.sqrt(VarMCO2[1, 1])
        
        AutoCov2 = np.array([calculate_autocov(Resid2, j, T) for j in range(q + 1)])
        
        aux2 = np.zeros(q)
        for j in range(1, q + 1):
            aux2[j-1] = (1 - (j / (q + 1))) * AutoCov2[j-1]
        
        lambda2_2 = np.zeros(q)
        for j in range(1, q + 1):
            lambda2_2[j-1] = AutoCov2[0] + 2 * np.sum(aux2[0:j])
        
        Zrho2 = T * (rho2 - 1) - 0.5 * ((T**2 * VarMCO2[1, 1]) / s2_2) * (lambda2_2 - AutoCov2[0])
        
        for i in range(q):
            resultados.append({'Metrica': f'Zrho Case2 (q={i+1})', 'Valor': Zrho2[i]})
            
    # --- Caso 4: Con constante y tendencia ---
    if regression in ['case4', 'all'] and Beta4 is not None:
        rho4 = Beta4[1]
        Resid4 = Y - X4 @ Beta4
        k4 = X4.shape[1]
        s2_4 = (Resid4.T @ Resid4) / (T - k4)
        VarMCO4 = s2_4 * np.linalg.inv(X4.T @ X4)
        sigmarho4 = np.sqrt(VarMCO4[1, 1])
        
        AutoCov4 = np.array([calculate_autocov(Resid4, j, T) for j in range(q + 1)])
        
        aux4 = np.zeros(q)
        for j in range(1, q + 1):
            aux4[j-1] = (1 - (j / (q + 1))) * AutoCov4[j-1]

        lambda2_4 = np.zeros(q)
        for j in range(1, q + 1):
            lambda2_4[j-1] = AutoCov4[0] + 2 * np.sum(aux4[0:j])
            
        Zrho4 = T * (rho4 - 1) - 0.5 * ((T**2 * VarMCO4[1, 1]) / s2_4) * (lambda2_4 - AutoCov4[0])
        Zt4 = ((rho4 - 1) / sigmarho4) * np.sqrt(AutoCov4[0] / lambda2_4) - \
              0.5 * (lambda2_4 - AutoCov4[0]) / np.sqrt(lambda2_4) * (T * sigmarho4 / np.sqrt(s2_4))
        
        for i in range(q):
            resultados.append({'Metrica': f'Zrho Case4 (q={i+1})', 'Valor': Zrho4[i]})
            resultados.append({'Metrica': f'Zt Case4 (q={i+1})', 'Valor': Zt4[i]})
            
    return pd.DataFrame(resultados)
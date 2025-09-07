import numpy as np
import pandas as pd
from scipy import stats

def test_gls(A, d=2, p=1, alpha=0.05):
    """
    Test ADF-GLS que retorna resultados en formato de tabla.
    
    Parámetros:
    -----------
    A : np.ndarray
        Serie de tiempo a analizar.
    d : int, opcional
        Componente determinístico (default=2):
        2 = solo intercepto.
        3 = intercepto y tendencia.
    p : int, opcional
        Orden autorregresivo del test.
    alpha : float, opcional
        Nivel de significancia (default=0.05).
        
    Retorna:
    --------
    pandas.DataFrame con los resultados del test.
    """
    
    A = np.array(A).flatten()
    T = len(A)
    
    # Validaciones
    if T < 20:
        raise ValueError("La serie es demasiado corta para el test ADF-GLS")
    if p >= T - d: # Corregido para evitar matriz singular
        raise ValueError("El orden p es demasiado grande para el largo de la serie")
    if alpha not in [0.01, 0.05, 0.10]:
        # Se puede relajar esta restricción ahora que el p-valor es numérico
        print(f"Advertencia: alpha={alpha} no es un nivel estándar, la decisión se basará en el p-valor numérico.")

    # --- 1. Des-tendenciación GLS ---
    if d == 2:
        c = -7.0
        z = np.ones((T, 1))
        tipo_deterministico = "Intercepto"
    else: # d == 3
        c = -13.5
        z = np.array([[1, t] for t in range(1, T + 1)])
        tipo_deterministico = "Intercepto + Tendencia"

    # Construcción de las series transformadas
    A_gls = np.concatenate([[A[0]], A[1:] - (1 + c/T) * A[:-1]])
    z_gls = np.vstack([z[0], z[1:] - (1 + c/T) * z[:-1]])
    
    # Regresión GLS para obtener los coeficientes de la tendencia
    psi = np.linalg.inv(z_gls.T @ z_gls) @ (z_gls.T @ A_gls)
    
    # Serie des-tendenciada
    A_detrended = A - (z @ psi)

    # --- 2. Regresión ADF sobre la serie des-tendenciada ---
    delta_A = np.diff(A_detrended)
    A_lag = A_detrended[:-1]
    
    # Construcción de la matriz de regresión (X)
    n_reg = len(delta_A)
    X = np.ones((n_reg, p + 1))
    X[:, 0] = A_lag
    
    for j in range(p):
        X[:, j+1] = np.diff(A_detrended, n=1, prepend=np.nan)[j:j+n_reg]

    # Eliminar filas con NaN creadas por los rezagos
    X_valid = X[p:, :]
    delta_A_valid = delta_A[p:]
    
    # Estimación de la regresión ADF
    phi = np.linalg.lstsq(X_valid, delta_A_valid, rcond=None)[0]
    residuals = delta_A_valid - X_valid @ phi
    s2 = np.sum(residuals**2) / (len(delta_A_valid) - X_valid.shape[1])
    
    try:
        XTX_inv = np.linalg.inv(X_valid.T @ X_valid)
        V = s2 * XTX_inv
        adfgls_stat = phi[0] / np.sqrt(V[0, 0])
    except np.linalg.LinAlgError:
        # En caso de multicolinealidad, el test no es válido
        return pd.DataFrame({
            'Caso': [tipo_deterministico], 'Estadístico ADF-GLS': [np.nan],
            'P-valor (interpolado)': ['Error'], 'Decisión': ['Error de cálculo'],
            'Interpretación': ['Matriz singular, posible multicolinealidad.']
        }).set_index('Caso')
    
    if d == 2: # Solo intercepto
        criticos = {'1%': -2.57, '5%': -1.94, '10%': -1.62}
    else: # Intercepto y tendencia
        criticos = {'1%': -3.48, '5%': -2.89, '10%': -2.57}
    
    # Puntos para la interpolación (valores críticos y niveles de significancia)
    critical_points = np.array(list(criticos.values()))
    p_levels = np.array([0.01, 0.05, 0.10])
    
    # Calcular p-valor usando interpolación lineal
    p_value = np.interp(adfgls_stat, critical_points, p_levels, right=1.0, left=0.01)
    
    return pd.DataFrame({
        'model': [tipo_deterministico], 
        'statistic': [adfgls_stat],
        'p-value': [p_value],
    })
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
    if p >= T:
        raise ValueError("El orden p no puede ser mayor o igual al largo de la serie")
    if alpha not in [0.01, 0.05, 0.10]:
        raise ValueError("El nivel de significancia (alpha) debe ser 0.01, 0.05, o 0.10.")

    # --- Cálculos del test GLS ---
    if d == 2:
        c = -7.0
        z = np.ones((T, 1))
        zgls = np.vstack([[1], z[1:] - (1 + c/T) * z[:-1]])
        tipo_deterministico = "Intercepto"
    else:
        c = -13.5
        z = np.array([[1, t] for t in range(1, T + 1)])
        zgls = np.vstack([[1, 1], z[1:] - (1 + c/T) * z[:-1]])
        tipo_deterministico = "Intercepto + Tendencia"

    ygls = np.concatenate([[A[0]], A[1:] - (1 + c/T) * A[:-1]])
    psi = np.linalg.inv(zgls.T @ zgls) @ (zgls.T @ ygls)
    yadf = A - (z @ psi)

    # --- Construcción de la matriz X ---
    y_diff = np.diff(yadf)
    
    X = np.ones((T-1, p+1))
    X[:, 0] = yadf[:-1]
    
    for j in range(1, p+1):
        if j == 1:
            X[:, j] = np.r_[np.nan, y_diff[:-1]]
        else:
            X[:, j] = np.r_[np.full(j, np.nan), y_diff[:-(j)]]
    
    # Eliminar filas con NaN
    y_diff_valid = y_diff[p:]
    X_valid = X[p:, :]
    
    # --- Estimación de la regresión ADF ---
    phi = np.linalg.lstsq(X_valid, y_diff_valid, rcond=None)[0]
    residuals = y_diff_valid - X_valid @ phi
    s2 = np.sum(residuals**2) / (len(y_diff_valid) - X_valid.shape[1])
    
    XTX_inv = np.linalg.inv(X_valid.T @ X_valid)
    V = s2 * XTX_inv
    
    adfgls_stat = phi[0] / np.sqrt(V[0, 0])
    
    # --- Cálculo del p-valor ---
    p_value = stats.norm.cdf(adfgls_stat)

    # --- Valores críticos y decisión ---
    if d == 2:
        criticos = {'1%': -2.57, '5%': -1.94, '10%': -1.62}
    else:
        criticos = {'1%': -3.48, '5%': -2.89, '10%': -2.57}
    
    clave_alpha = f'{int(alpha*100)}%'
    valor_critico = criticos[clave_alpha]
    rechazo_h0 = adfgls_stat < valor_critico
    
    # --- Interpretación ---
    if rechazo_h0:
        interpretacion = "La serie es estacionaria"
    else:
        interpretacion = "La serie no es estacionaria"

    # --- Crear tabla de resultados ---
    resultados = pd.DataFrame({
        'Parametro': [
            'Estadistico ADF-GLS',
            'Valor critico (' + clave_alpha + ')',
            'p-valor (aproximado)',
            'Decision',
            'Interpretacion'
        ],
        'Valor': [
            f"{adfgls_stat:.4f}",
            f"{valor_critico:.4f}",
            f"{p_value:.4f}",
            "Rechazar H0" if rechazo_h0 else "No rechazar H0",
            interpretacion
        ]
    })
    
    return resultados
import numpy as np
import pandas as pd

def test_gls(A, d=2, k=1):
    """
    Implementación del test ADF-GLS que incluye una conclusión automática
    para decidir sobre la estacionariedad.
    
    Parámetros:
    -----------
    A : np.ndarray
        Serie de tiempo a analizar.
    d : int, opcional
        Componente determinístico (default=2):
        2 = solo intercepto.
        3 = intercepto y tendencia.
    k : int, opcional
        Número de rezagos para la corrección de autocorrelación (default=1).
        
    Retorna:
    --------
    pd.DataFrame con los resultados y la conclusión del test.
    """
    
    A = np.array(A).flatten()
    T = len(A)
    
    # --- Validación de Entradas ---
    if d not in [2, 3]:
        raise ValueError('Opción no válida: d debe ser 2 (intercepto) o 3 (intercepto + tendencia)')
    
    # --- Contenedor de Resultados ---
    resultados = {}

    # =========================================================================
    # PASO 1: Eliminación de Tendencia con Mínimos Cuadrados Generalizados (GLS)
    # =========================================================================
    if d == 2:
        c = -7.0
        z = np.ones((T, 1))
        zgls = np.vstack([[1], z[1:] - (1 + c/T) * z[:-1]])
    else:
        c = -13.5
        z = np.array([[1, t] for t in range(1, T + 1)])
        zgls = np.vstack([[1, 1], z[1:] - (1 + c/T) * z[:-1]])

    ygls = np.concatenate([[A[0]], A[1:] - (1 + c/T) * A[:-1]])
    psi = np.linalg.inv(zgls.T @ zgls) @ (zgls.T @ ygls)
    yadf = A - (z @ psi)

    # ============================================================================
    # PASO 2: Regresión de Dickey-Fuller Aumentado (ADF) sobre la serie sin tendencia
    # ============================================================================
    y_diff = yadf[k:] - yadf[k-1:-1]
    X = np.zeros((T - k, k))
    X[:, 0] = yadf[k-1:-1]
    for j in range(1, k):
        X[:, j] = (yadf[k-j-1:-j-1] - yadf[k-j-2:-j-2])
        
    phi = np.linalg.inv(X.T @ X) @ (X.T @ y_diff)
    e = y_diff - (X @ phi)
    s2 = (e.T @ e) / (X.shape[0] - X.shape[1])
    V = s2 * np.linalg.inv(X.T @ X)

    # =========================================================================
    # PASO 3: Cálculo del Estadístico y Valores Críticos
    # =========================================================================
    adfgls_stat = phi[0] / np.sqrt(V[0, 0])
    
    resultados['Estadistico'] = adfgls_stat
    resultados['Valor P'] = 'No calculado'
    
    if d == 2:
        criticos = {'1%': -2.57, '5%': -1.94, '10%': -1.62}
    else:
        criticos = {'1%': -3.48, '5%': -2.89, '10%': -2.57}
        
    resultados['Valor Critico 1%'] = criticos['1%']
    resultados['Valor Critico 5%'] = criticos['5%']
    resultados['Valor Critico 10%'] = criticos['10%']
    
    # =========================================================================
    # NUEVO: PASO 4 - Decisión y Conclusión Automática
    # =========================================================================
    # Esta es la línea clave que permite tomar la decisión.
    # Compara el estadístico con el valor crítico al 5% de significancia.
    
    if adfgls_stat < criticos['5%']:
        conclusion = "El estadístico es MENOR que el valor crítico al 5%. Se rechaza H₀. La serie es ESTACIONARIA."
    else:
        conclusion = "El estadístico es MAYOR que el valor crítico al 5%. No se puede rechazar H₀. La serie tiene RAÍZ UNITARIA (no es estacionaria)."
    
    resultados['Conclusion (al 5%)'] = conclusion
    
    # --- Información Adicional ---
    resultados['Componente deterministico'] = 'Intercepto' if d == 2 else 'Intercepto y Tendencia'
    resultados['Numero de rezagos'] = k-1
    resultados['Observaciones'] = T
    
    # --- Formateo Final ---
    return pd.DataFrame(list(resultados.items()), columns=['Metrica', 'Valor'])

# --- Ejemplo de uso ---
# np.random.seed(123)
# serie_no_estacionaria = np.random.randn(100).cumsum() + 50
# serie_estacionaria = np.random.randn(100)

# print("--- Test para serie NO estacionaria ---")
# resultados1 = test_gls(serie_no_estacionaria, d=2, k=4)
# print(resultados1)
# print("\n--- Test para serie ESTACIONARIA ---")
# resultados2 = test_gls(serie_estacionaria, d=2, k=4)
# print(resultados2)
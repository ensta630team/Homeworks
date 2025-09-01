import numpy as np 
import sympy as sp

from scipy.linalg import lu


def calculate_irf(F, H, phi=None, method='exact'):
    """
    Calcula la Función de Impulso-Respuesta (IRF) para un horizonte H.
    Esta funcion esta hecha para una matriz F de un modelo univariado
    Argumentos:
        F (np.ndarray): La matriz acompañante del proceso.
        H (int): El horizonte (número de períodos) para el cálculo.
        phi (np.ndarray, opcional): Coeficientes AR, necesarios para el método 'simulation'.
        method (str, opcional): El método de cálculo a utilizar. Opciones:
            - 'exact': Usa descomposición de eigenvalores (rápido, requiere matriz diagonalizable).
            - 'exact_old': Usa potencias de la matriz (más lento pero general).
            - 'jordan': Usa la forma normal de Jordan (para matrices no diagonalizables).
            - 'simulation': Calcula la IRF de forma recursiva.

    Retorna:
        np.ndarray: Un array con los valores de la IRF para H períodos.
    """
    p = F.shape[-1]
    irf_values = np.zeros(H)
    irf_values[0] = 1 # El impulso inicial es 1 por definición.

    eigenvals, eigenvec = np.linalg.eig(F)
    
    # Verifica si hay eigenvalores repetidos, lo que puede requerir la forma de Jordan.
    mult = len(set(eigenvals))
    if len(eigenvals) != mult or method == 'jordan':
        # Para matrices no diagonalizables, se usa la forma de Jordan con sympy.
        Fsymp = sp.Matrix(F)
        P, J = Fsymp.jordan_form()
        print("Matriz de Jordan (J):", J)

    elif method == 'exact':
        # Método eficiente usando descomposición de eigenvalores: F^h = P * D^h * P^-1
        for h in range(1, H):
            temp = np.eye(p) * np.power(eigenvals, h)
            temp = np.matmul(eigenvec, temp)
            temp = np.matmul(temp, np.linalg.inv(eigenvec))
            irf_values[h] = temp[0, 0]

    elif method == 'exact_old':
        # Método directo calculando la potencia de la matriz F en cada paso.
        for h in range(1, H):
            F_h = np.linalg.matrix_power(F, h)
            irf_values[h] = F_h[0, 0]

    elif method == 'simulation':
        # Calcula la IRF de forma recursiva, como si fuera una simulación.
        for h in range(1, H):
            past_values = irf_values[max(0, h - p):h]
            coeffs_to_use = phi[:len(past_values)]
            irf_values[h] = np.dot(coeffs_to_use, past_values[::-1])
    
    return irf_values


# ============================================================================
# ====================  ~~~~VAR~~~~ ==========================================
# ============================================================================
def cholesky_irf(psi, omega, H):
    """
    Calcula la Función de Impulso-Respuesta (IRF) para H periodos.
    
    Argumentos:
        H (int): El horizonte de tiempo para el cual calcular la IRF.

    Retorna:
        np.ndarray: Un array de forma (H, n, n) donde n es el inp_dim.
                    Cada matriz IRF[s, :, :] es la respuesta en el tiempo s.
    """
    
    # Descomponer triangularmente la matriz de covarianzas
    # Usamos la descomposición de Cholesky que es más estándar para matrices simétricas.
    # Es numéricamente más estable y garantiza una matriz triangular inferior única.
    try:
        L = np.linalg.cholesky(omega)
    except np.linalg.LinAlgError:
        # Si falla Cholesky (no es pos-def), usamos LU
        _, L, _ = lu(omega)
    
    # Calcular la IRF para cada horizonte s
    irf_list = [psi @ L for psi in psi]
    
    # Convertir a un único array de numpy y devolver
    return np.array(irf_list)


def generalized_irf(psi: list, omega: np.ndarray, H: int) -> np.ndarray:
    """
    Calcula la IRF Generalizada usando la secuencia de matrices Psi.
    """
    n = omega.shape[0]
    H = len(psi) - 1
    
    # Preparamos el array final para almacenar todas las respuestas
    irf_results = np.zeros((H + 1, n, n))
    
    # Obtenemos las desviaciones estándar de los errores (denominador)
    sigma_jj = np.sqrt(np.diag(omega))

    # Bucle para cada horizonte de tiempo h
    for h, Psi_h in enumerate(psi):
        # Bucle para cada shock j (columna de la matriz de IRF)
        for j in range(n):
            # Numerador: Psi_h * omega_j (columna j de la matriz de covarianza)
            numerador = Psi_h @ omega[:, j]
            
            # La respuesta de todas las variables al shock j en el horizonte h
            irf_results[h, :, j] = numerador / sigma_jj[j]

    return irf_results
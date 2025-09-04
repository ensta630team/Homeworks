from scipy.stats import f as f_distribution
import numpy as np


def test_chow(model, X, y_true, quiebre: int):
    """
    Realiza el Test de Chow para evaluar un cambio estructural en un punto específico.

    Argumentos:
        model: Un objeto de modelo ya ajustado (debe tener un método .predict()).
        X (np.ndarray): La matriz completa de regresores.
        y_true (np.ndarray): El vector completo de la variable dependiente.
        quiebre (int): El índice de la observación donde se prueba el cambio estructural.

    Retorna:
        tuple: Una tupla conteniendo:
            - chow_stat (float): El estadístico F del test de Chow.
            - p_valor (float): El p-valor asociado al estadístico.
    """
    n, k = X.shape

    # Suma de Cuadrados de los Residuos (SCR) del modelo completo (restringido)
    y_pred_full = model.predict(X)
    residuos_full = y_true - y_pred_full
    sc_restringido = np.sum(residuos_full**2)

    # SCR del modelo pre-quiebre
    X_pre, y_pre = X[0:quiebre], y_true[0:quiebre]
    
    # Estimar beta_pre por OLS
    beta_pre = np.linalg.inv(X_pre.T @ X_pre) @ X_pre.T @ y_pre
    residuos_pre = y_pre - (X_pre @ beta_pre)
    
    sc_pre = np.sum(residuos_pre**2)

    # 3. SCR del modelo post-quiebre
    X_post, y_post = X[quiebre:], y_true[quiebre:]
    # Estimar beta_post por OLS
    beta_post = np.linalg.inv(X_post.T @ X_post) @ X_post.T @ y_post
    residuos_post = y_post - (X_post @ beta_post)
    sc_post = np.sum(residuos_post**2)

    # Suma de SCR del modelo sin restringir (dos regresiones)
    sc_sin_restringir = sc_pre + sc_post

    # 4. Cálculo del estadístico F de Chow
    numerador = (sc_restringido - sc_sin_restringir) / k
    denominador = sc_sin_restringir / (n - 2 * k)
    
    # Manejar el caso de denominador cero para evitar errores
    if denominador == 0:
        return np.inf, 0.0

    chow_stat = numerador / denominador
    
    # 5. Cálculo del p-valor
    # El estadístico sigue una distribución F con (k, n - 2k) grados de libertad.
    p_valor = 1 - f_distribution.cdf(chow_stat, dfn=k, dfd=n - 2 * k)

    return chow_stat, p_valor
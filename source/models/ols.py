import numpy as np
import statsmodels.api as sm


class OLS:
    """
    Clase para ajustar un modelo de regresión lineal por Mínimos Cuadrados Ordinarios (OLS).
    """
    def __init__(self):
        self.beta = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo OLS a los datos proporcionados.

        Argumentos:
            X (np.ndarray): Matriz de regresores (variables independientes).
            y (np.ndarray): Vector de la variable dependiente.
        """
        # Guarda los datos como atributos de la instancia
        self.X = X
        self.y = y
        
        # Calcula los coeficientes beta
        try:
            XX_inv = np.linalg.inv(X.T @ X)
            self.beta = XX_inv @ X.T @ y
        except np.linalg.LinAlgError:
            self.beta = np.full(X.shape[1], np.nan)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando los coeficientes ajustados del modelo.
        """
        if self.beta is None:
            raise ValueError("El modelo debe ser ajustado primero con el método .fit()")
        
        return X @ self.beta
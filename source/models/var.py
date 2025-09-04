import numpy as np 

from scipy.linalg import solve_discrete_lyapunov

from source.data.sampling import generate_stationary_var
from source.data.transform import create_var_dataset
from source.models.ols import OLS
from source.fit.error import get_var_standard_errors
from source.statistics import impulse_response as irf

class VAR:

    def __init__(self,  
                inp_dim=1,
                p=1,
                c=0, 
                sigma=0.1, # solo para muestrear
                params_distribution: str ="stationary",
                **kwargs):

        self.inp_dim   = inp_dim
        self.p         = p
        self.residuals = None # Esto es para guardar los residuos despues de ajustar con fit
        self.y         = None
        # Estandarizar 'c' y 'sigma' para que siempre sean vectores
        if isinstance(c, (int, float)):
            self.c = np.ones(inp_dim) * c
        else:
            self.c = c
            
        if isinstance(sigma, (int, float)):
            # Si sigma es un numero, conviertelo en un vector con ese valor repetido
            self.sigma = np.ones(inp_dim) * sigma
        else:
            self.sigma = sigma
        
        self.omega_hat = np.diag(self.sigma**2) # Si no hay modelo ajustado se asume sigma
        self._is_stationary_cached = None

        if params_distribution=='stationary':
            # Si se pide inicialización aleatoria, usamos la función correcta
            phi_gen, c_gen, _ = generate_stationary_var(n=inp_dim, p=p)
            self.phi = phi_gen
            self.c   = c_gen
        else:
            self.phi = np.zeros((p, inp_dim, inp_dim))
        
        # El parámetro pi se construye a partir de phi y c
        self.pi = self._build_Pi_matrix()

    def get_unconditional_mean(self) -> np.ndarray:
        """
        Calcula la media incondicional (de largo plazo) del proceso.
        Solo es válido si el proceso es estacionario.
        """
        if not self._is_stationary():
            _, ev = self._check_stationarity()
            raise ValueError("La media incondicional no está definida para un proceso no estacionario.")
        
        if self.c is None or self.phi is None:
            raise ValueError("El modelo debe ser ajustado primero con el método .fit().")

        # Sumamos todas las matrices de coeficientes Phi
        sum_of_phis = np.sum(self.phi, axis=0)
        # Creamos la matriz identidad del tamaño adecuado (n x n)
        identity_matrix = np.identity(self.inp_dim)
        # Calculamos (I - sum(Phi))
        matrix_to_invert = identity_matrix - sum_of_phis
        # Invertimos la matriz y la multiplicamos por el vector de constantes
        inv_matrix = np.linalg.inv(matrix_to_invert)
        unconditional_mean = inv_matrix @ self.c
        return unconditional_mean

    def get_unconditional_std(self) -> np.ndarray:
        """
        Calcula la desviación estándar incondicional del proceso.
        Solo es válido si el proceso es estacionario.
        """
        if not self._is_stationary():
            raise ValueError("La desviación estándar incondicional no está definida para un proceso no estacionario.")
            
        if self.omega_hat is None:
            raise ValueError("El modelo debe ser ajustado primero con el método .fit() para estimar omega_hat.")

        # 1. Obtenemos la matriz compañera F
        F = self._build_F_matrix()
        
        # 2. Creamos la matriz de covarianza de errores aumentada
        # Es una matriz grande de ceros con omega_hat en la esquina superior izquierda
        n_stacked = self.inp_dim * self.p
        omega_augmented = np.zeros((n_stacked, n_stacked))
        omega_augmented[:self.inp_dim, :self.inp_dim] = self.omega_hat
        
        # 3. Resolvemos la ecuación de Lyapunov discreta para la matriz de covarianza del proceso apilado
        # La ecuación es: Gamma_stacked = F @ Gamma_stacked @ F.T + Omega_augmented
        gamma_stacked = solve_discrete_lyapunov(F, omega_augmented)
        
        # 4. La varianza incondicional de nuestro proceso Y_t es el bloque superior izquierdo (n x n)
        gamma_0 = gamma_stacked[:self.inp_dim, :self.inp_dim]
        
        # 5. La desviación estándar es la raíz cuadrada de la diagonal (las varianzas)
        unconditional_std = np.sqrt(np.diag(gamma_0))
        
        return unconditional_std

    def get_irf(self, H: int = 20, method='cholesky') -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF) para H periodos.
        
        Argumentos:
            H (int): El horizonte de tiempo para el cual calcular la IRF.

        Retorna:
            np.ndarray: Un array de forma (H, n, n) donde n es el inp_dim.
                        Cada matriz IRF[s, :, :] es la respuesta en el tiempo s.
        """
        if self.phi is None or self.omega_hat is None:
            raise ValueError("El modelo debe ser ajustado primero con el método .fit().")
        
        # Calcular la secuencia de matrices Psi
        # H+1 porque la secuencia incluye el término 0
        psi_sequence = self._compute_psi_sequence(H=H + 1)
        
        # Descomponer triangularmente la matriz de covarianzas
        if method == 'cholesky':
            irf_list = irf.cholesky_irf(psi_sequence, self.omega_hat, H)
        elif method == 'girf':
            assert self.omega_hat.shape[0] == self.inp_dim
            irf_list = irf.generalized_irf(psi_sequence, self.omega_hat, H)
        # OJO! Aqui se pueden seguir agregando metodos con "elif"
        else:
            print('Ese metodo no esta implementado jeje')
 
    
        return irf_list


    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 0) -> np.ndarray:
        """
        Genera una muestra simulada de la serie de tiempo del proceso VAR.

        Argumentos:
            n_samples (int): El número de observaciones a generar y devolver.
            initial_values (np.ndarray, opcional): Un array de forma (p, n) con los
                valores iniciales para arrancar el proceso. Si es None, se usan ceros.
            burn_in (int): El número de muestras iniciales a generar y descartar
                para mitigar el efecto de los valores iniciales.

        Retorna:
            np.ndarray: Un array de forma (n_samples, n) con la serie de tiempo generada.
        """
        if self.c is None or self.phi is None:
            raise ValueError("Los parámetros del modelo (c, phi) no están definidos.")
        
        # Determinar la matriz de covarianzas del error (Omega) ---
        if self.omega_hat is not None:
            # Si el modelo fue ajustado, usamos la matriz estimada
            omega = self.omega_hat
        elif self.sigma is not None:
            # Si no, creamos una a partir de sigma (asumiendo errores no correlacionados)
            omega = np.diag(self.sigma**2)
        else:
            raise ValueError("Se necesita 'omega_hat' (de .fit()) o 'sigma' (de __init__) para generar muestras.")

        total_samples = n_samples + burn_in
        
        # Template para la serie de tiempo que vamos a generar
        samples = np.zeros((total_samples, self.inp_dim))
        
        if initial_values is not None:
            if initial_values.shape != (self.p, self.inp_dim):
                raise ValueError(f"initial_values debe tener forma ({self.p}, {self.inp_dim})")
            samples[:self.p] = initial_values
        # Si no se proveen valores iniciales, se queda con los p vectores de ceros.
        
        # Generar todos los shocks aleatorios de una vez
        mean = np.zeros(self.inp_dim)
        shocks = np.random.multivariate_normal(mean=mean, cov=omega, size=total_samples)
        
        # Iterar para generar la serie de tiempo ---
        for t in range(self.p, total_samples):
            # Inicializamos y_t con el intercepto
            y_t = self.c.copy()
            
            # Anadimos los términos de los rezagos
            for i in range(self.p):
                # phi[i] es la matriz Φ_{i+1}
                # samples[t-1-i] es el vector Y_{t-(i+1)}
                y_t += self.phi[i] @ samples[t - 1 - i]
                
            # Anadimos el shock aleatorio del período t
            y_t += shocks[t]
            
            # Guardamos el valor generado
            samples[t] = y_t
            
        # Descartar el período de burn-in y devolver ---
        return samples[burn_in:]
    
    def simulate(self, T: int, initial_values: np.ndarray = None):
        """
        Simula una trayectoria de T observaciones a partir de los parámetros ajustados del VAR.

        Args:
            T (int): El número de observaciones a simular.
            initial_values (np.ndarray, optional): Los valores iniciales para los rezagos. 
                                                    Si es None, se usarán ceros.
        
        Returns:
            np.ndarray: La serie de tiempo simulada de tamaño (T, n_vars).
        """
        if self.phi is None:
            raise RuntimeError("El modelo debe ser ajustado primero con .fit()")

        p, n_vars, _ = self.phi.shape
        
        # Usar los últimos p valores de los datos originales si no se proveen valores iniciales
        if initial_values is None:
            initial_values = self.y[-p:]

        # Generar errores/innovaciones
        # Cholesky de la matriz de covarianza de residuos para generar errores correlacionados
        L = np.linalg.cholesky(self.omega_hat) 
        innovations = (L @ np.random.randn(n_vars, T)).T

        # Almacenar resultados
        y_simulated = np.zeros((T, n_vars))
        y_history = list(initial_values)
        phi_pred_matrix = self.phi.transpose(1, 0, 2).reshape(n_vars, p * n_vars)
        
        for t in range(T):
            # Construir el vector de rezagos Y_{t-1}, ..., Y_{t-p}
            y_lags_flat = np.array(y_history[::-1]).flatten()
            
            # Calcular el valor predicho
            y_t = self.c + phi_pred_matrix @ y_lags_flat + innovations[t]
                
            y_simulated[t, :] = y_t
            
            # Actualizar el historial
            y_history.append(y_t)
            y_history.pop(0)

        return y_simulated

    def fit(self, X):
        # Creamos la matriz de predictores con "p" rezagos
        X_ols, y_ols = create_var_dataset(X, lag=self.p, add_intercept=True)
        T_effective = X_ols.shape[0]
        
        # Ajustamos el modelo y guardamos la matriz de coeficientes completa
        self.pi = np.linalg.inv(X_ols.T @ X_ols) @ (X_ols.T @ y_ols)

        # La primera fila de pi es el vector de constantes (intercepto)
        self.c = self.pi[0, :]
        
        # Las filas restantes son las matrices phi_1, phi_2, ..., phi_p apiladas
        phi_stacked = self.pi[1:, :]

        # Reorganizamos este vector largo en la estructura (p, n, n) de self.phi
        self.phi = np.array([phi_stacked[i*self.inp_dim:(i+1)*self.inp_dim, :].T for i in range(self.p)])
        
        # Calculamos los residuos
        y_pred = self.predict(X_ols)
        residuals_matrix = y_ols - y_pred
        
        # Guardamos estos valores para futuros calculos
        self.residuals = residuals_matrix
        self.y = y_ols
        
        # Usar las observaciones efectivas en el denominador
        self.omega_hat = (residuals_matrix.T @ residuals_matrix) / T_effective
        
        # errores estándar
        std_errors = get_var_standard_errors(X_ols, y_ols, y_pred, white=False)
        
        # Forzamos la re-evaluación de la estacionariedad con los nuevos coeficientes
        self._is_stationary_cached = None 
        
        return std_errors

    def predict(self, X):
        if self.pi is None:
            raise ValueError("El modelo debe ser ajustado primero con el método .fit()")
        return X @ self.pi

    # ===================================================================================
    # ====== METODOS PRIVADOS ===========================================================
    # ===================================================================================
    def _decompose_matrix(self, H=5):
        psi = self._compute_psi_sequence(H+1)
        M = irf.compute_variance_decomposition(psi, self.omega_hat, H)
        return M
    
    def _compute_psi_sequence(self, H: int) -> list:
        """
        Calcula la sucesión de matrices Ψ_i para i = 0, ..., H-1.
        Esta es la representación VMA(inf) del proceso VAR.
        """
        if len(self.phi) != self.p:
            raise ValueError(f"Se esperaban {self.p} matrices Φ, pero se recibieron {len(self.phi)}")
        
        n = self.inp_dim
        
        # Inicializar la lista de Ψ
        Psi = []
        
        # Ψ₀ = Id (matriz identidad)
        Psi0 = np.eye(n)
        Psi.append(Psi0)
        
        # Calcular los términos restantes
        for s in range(1, H):
            suma = np.zeros((n, n))
            # Calcular: Ψ_s = sum_{i=1}^p Φ_i * Ψ_{s-i}
            for i in range(1, self.p + 1):
                if s - i >= 0:  
                    suma += np.dot(self.phi[i-1], Psi[s-i])
            Psi.append(suma)
        return Psi
    
    def _is_stationary(self) -> bool:
        """
        Verifica si el proceso es estacionario usando un valor en caché.
        """
        self._is_stationary_cached, ev = self._check_stationarity()
        return self._is_stationary_cached
    
    def _check_stationarity(self) -> bool:
        """
        Revisa si los eigenvalores de la matriz F son menores a 1 en módulo.
        """
        if self.p == 0:
            return True
            
        F = self._build_F_matrix()
        eigenvalues = np.linalg.eigvals(F)
        self._is_stationary_cached = np.all(np.abs(eigenvalues) < 1)
        
        return self._is_stationary_cached, eigenvalues
    
    def _build_Pi_matrix(self):
        if self.phi is None or self.c is None:
            return None
            
        # Transponemos cada matriz phi antes de apilar
        phi_transposed_and_stacked = np.vstack([phi_i.T for phi_i in self.phi])

        # Añadimos el intercepto como la primera fila
        Pi = np.vstack([self.c, phi_transposed_and_stacked])
        return Pi
    
    def _build_F_matrix(self):
        if self.p == 0:
            return np.empty((0, 0))
        I = np.identity(self.inp_dim*(self.p-1))
        zeros = np.zeros([self.inp_dim*self.p-self.inp_dim, 
                          self.inp_dim])
        top    = np.hstack(self.phi)
        bottom = np.hstack([I, zeros])
        F = np.vstack([top, bottom])
        return F

    # ==================================================================================
    # ====== PROPODIEDADES =============================================================
    # ==================================================================================
    @property
    def order(self):
        return self.p
    
    # ==================================================================================
    # ==================================================================================
    # ==================================================================================
    def __str__(self):
        """
        Representación en string del modelo VAR, mostrando un resumen de sus
        parámetros y propiedades.
        """
        header = f"{'='*60}\n"
        header += f"{'Vector Autoregression (VAR) Model Summary':^60}\n"
        header += f"{'='*60}\n"

        # --- Propiedades básicas ---
        info = f"VAR(p={self.p}, n={self.inp_dim})\n"
        
        # --- Estacionariedad ---
        try:
            is_stationary, eigenvalues = self._check_stationarity()
            max_eig = np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0
            info += f"Estacionario: {'Sí' if is_stationary else 'No'} (Max Eigenvalue: {max_eig:.4f})\n"
        except Exception:
            info += "Estacionario: No determinado\n"
        
        info += f"{'-'*60}\n"

        # --- Propiedades de Largo Plazo ---
        info += f"{'Propiedades de Largo Plazo':^60}\n"
        # Media incondicional
        try:
            mean = self.get_unconditional_mean()
            info += f"Media Incondicional:\n{np.array2string(mean, precision=4)}\n\n"
        except (ValueError, np.linalg.LinAlgError) as e:
            info += f"Media Incondicional: No definida\n\n"
            
        # Desviación estándar incondicional
        try:
            std = self.get_unconditional_std()
            info += f"D.E. Incondicional:\n{np.array2string(std, precision=4)}\n"
        except (ValueError, np.linalg.LinAlgError) as e:
            info += f"D.E. Incondicional: No definida\n"
            
        info += f"{'-'*60}\n"

        # --- Coeficientes ---
        info += f"{'Coeficientes del Modelo':^60}\n"
        # Intercepto (c)
        if self.c is not None:
            info += f"Intercepto (c):\n{np.array2string(self.c, precision=4, suppress_small=True)}\n\n"
        else:
            info += "Intercepto (c): No definido\n\n"
            
        # Matrices Phi
        if self.phi is not None and len(self.phi) > 0:
            for i, phi_matrix in enumerate(self.phi):
                info += f"Matriz Phi_{i+1}:{phi_matrix.shape}\n"
        else:
            info += "Matrices Phi: No definidas\n\n"
        
        info += f"{'-'*60}\n"

        # --- Matriz de Covarianza de Residuos ---
        info += f"{'Matriz de Covarianza de Residuos (Omega)':^60}\n"
        if hasattr(self, 'omega_hat') and self.omega_hat is not None:
            info += f"Omega: {self.omega_hat.shape}\n"
        else:
            info += "No definida (el modelo no ha sido ajustado)\n"

        footer = f"{'='*60}\n"
        
        return header + info + footer
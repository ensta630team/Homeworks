# Importa estas librerías al inicio de tu archivo ivar.py si no las tienes ya
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from source.models.var import VAR


class IVAR(VAR):
    def __init__(self,
                 inp_dim=1,
                 p=1,
                 c=0,
                 sigma=0.1,
                 params_distribution: str = "stationary",
                 **kwargs):
        super().__init__(inp_dim, p, c, sigma, params_distribution, **kwargs)

        # Initializamos para ambos
        self.B0 = None
        self.D  = None
    
    def set_B0(self, B0, X):
        self.B0 = B0
        _ = self.fit(X)         
        self.B0, self.D = self._estimar()

    def _estimar(self):
        # Crear matriz D
        D = sp.symbols(f'd1:{self.inp_dim+1}')
        D = sp.diag(*D)

        # Transformar matriz omega para operar en SymPy
        omega = sp.Matrix(self.omega_hat)
    
        # Usar la transpuesta de B0
        B0t = self.B0.T
        Di = D.inv()

        # Parte 1: 2 * log(det(B0))
        detB0 = self.B0.det()
        inicial = 2 * sp.log(detB0)

        # Parte 2: log(det(D))
        detD = D.det()
        medio = sp.log(detD)

        # Parte 3: trace(B0.T * D^-1 * B0 * Omega)
        final = B0t * Di * self.B0 * omega
        trazafinal = final.trace()

        # Función de verosimilitud (sin constantes aditivas)
        verosimilitud = inicial - medio - trazafinal

        # Variables a estimar
        variables = list(verosimilitud.free_symbols)

        # ===================================================
        # Maximizar la verosimilitud
        # ===================================================
        # Calcular el gradiente (la derivada respecto a cada variable)
        gradiente = [sp.diff(verosimilitud, var) for var in variables]
        
        # Convertir el gradiente simbólico a una función numérica rápida
        # Esta función tomará los valores de las variables y devolverá una lista de resultados
        gradiente_func = sp.lambdify(variables, gradiente, 'numpy')

        # Definir la función objetivo para el solucionador.
        # Esta función debe tomar un array de NumPy 'x' y devolver un array de NumPy.
        def objetivo(x):
            # Desempaquetamos 'x' para pasarlo a la función del gradiente
            resultado_lista = gradiente_func(*x)
            # Convertimos la lista de salida en un array de NumPy y lo aplanamos
            return np.array(resultado_lista, dtype=np.float64).flatten()

        # Establecer un punto de partida aleatorio para el solucionador
        guess = np.random.rand(len(variables))
        
        # 5. Resolver el sistema de ecuaciones: gradiente = 0
        try:
            estimacion_valores = fsolve(objetivo, guess)
        except Exception as e:
            print(f"Error durante la optimización con fsolve: {e}")
            print("El solucionador numérico no pudo converger. Intenta ejecutar de nuevo.")
            return None, None

        # Crear un diccionario con los resultados {variable: valor}
        resultados = {var: val for var, val in zip(variables, estimacion_valores)}
        
        # Sustituir los valores estimados en las matrices B0 y D
        B0_hat = self.B0.subs(resultados)
        D_hat  = D.subs(resultados)
        
        # Convertir a matrices de NumPy
        B0_hat = np.array(B0_hat, dtype=np.float64)
        D_hat  = np.array(D_hat, dtype=np.float64)
        
        print("Estimación finalizada con éxito.")
        return B0_hat, D_hat
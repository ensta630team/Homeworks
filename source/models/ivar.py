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

    def set_B0(self, B0, X, guess):
        self.B0 = B0
        _ = self.fit(X)
        guess = {'b2': 1, 'd2': 1/10000, 'd1': 1/100000, 'b1': 0.05, 'b3': 0.05, 'd3': 1/1000000}
        self.B0, self.D = self._estimar(guess)

    def _estimar(self,guessx):
        # Crear matriz D
        D = sp.symbols(f'd1:{self.inp_dim+1}')
        D = sp.diag(*D)

        # Transformar matriz omega para operar en SymPy
        omega = sp.Matrix(self.omega_hat)

        # La sutilizare
        B0t = self.B0.T
        Di = D.inv()
        # Primera expresión
        detB0 = self.B0.det()
        inicial = 2 * sp.log(detB0)
        # Segunda expresion
        detD = D.det()
        medio = sp.log(detD)
        # Expresion final
        final = B0t * Di * self.B0 * omega
        trazafinal = final.trace()
        # Verosimilitud completa
        # (self["t"]-self["p"])/2 *
        verosimilitud = (inicial - medio - trazafinal)
        # Variables
        variables = list(verosimilitud.free_symbols)

        variables_str = [str(x) for x in variables]
        guess = [guessx[x] for x in variables_str]
        # Maximizar verosimilitud
        # Buscar gradiente igual a 0
        gradiente =  [sp.diff(verosimilitud, x) for x in variables]
        # Resolver sistema de ecuaciones: gradiente = 0

        estimacion = sp.nsolve(gradiente, variables, guess)
        variables_IVAR = variables
        estimacion_IVAR = estimacion
        resultados = {llave: valor for llave,valor in zip(variables_IVAR,estimacion_IVAR)}
        # Crear a mano B0_hat y D_hat
        B0_hat = self.B0.subs(resultados)
        D_hat = D.subs(resultados)
        # Transformarlos a Numpy
        B0_hat = np.array(B0_hat,dtype=np.float64)
        D_hat = np.array(D_hat,dtype=np.float64)
        #print(B0_hat)
        #print(D_hat)
        # Añadirlos
        # Recrear
        B0_hati = np.linalg.inv(B0_hat)
        B0_hatit = B0_hati.T
        Omega_recrear = B0_hati @ D_hat @ B0_hatit
        # print(B0_hat)
        # print(D_hat)
        # print(omega)
        # print(Omega_recrear)
        return B0_hat, D_hat
import numpy as np
from numpy.polynomial.polynomial import polyfromroots
import statsmodels.api as sm


# Diccionario que mapea nombres de distribuciones a funciones lambda para generar parámetros aleatorios.
# Esto permite crear parámetros de forma flexible a partir de diferentes distribuciones de probabilidad.
DISTRIBUTION_GENERATORS = {
    'normal': lambda **kwargs: np.random.normal(
        loc=kwargs.get('mean', 0.0),      # Media por defecto = 0
        scale=kwargs.get('std', 1.0),       # Desv. Estándar por defecto = 1
        size=kwargs['p']),
    'exponential': lambda **kwargs: np.random.exponential(
        scale=kwargs.get('scale', 1.0),   # Escala por defecto = 1
        size=kwargs['p']),
    'uniform': lambda **kwargs: np.random.uniform(
        low=kwargs.get('low', 0.0),       # Límite inferior por defecto = 0
        high=kwargs.get('high', 1.0),     # Límite superior por defecto = 1
        size=kwargs['p']),
    'beta': lambda **kwargs: np.random.beta(
        a=kwargs.get('alpha', 1.0),       # Alpha por defecto = 1
        b=kwargs.get('beta', 1.0),        # Beta por defecto = 1
        size=kwargs['p']),
    'gamma': lambda **kwargs: np.random.gamma(
        shape=kwargs['shape'], 
        scale=kwargs.get('scale', 1.0), 
        size=kwargs['p']),
    'poisson': lambda **kwargs: np.random.poisson(
        lam=kwargs.get('lambda', 1.0), 
        size=kwargs['p'])
}

def generate_stationary_phi(p: int) -> np.ndarray:
    """
    Genera un conjunto de coeficientes phi para un proceso AR(p) que es estacionario por construcción.
    
    El método consiste en generar raíces aleatorias dentro del círculo unitario en el plano complejo
    y luego construir el polinomio característico correspondiente para derivar los coeficientes phi.
    """
    # Genera raíces complejas conjugadas para asegurar que los coeficientes del polinomio sean reales.
    num_complex_pairs = p // 2
    radii = np.sqrt(np.random.uniform(0, 1, size=num_complex_pairs))
    thetas = np.random.uniform(0, 2 * np.pi, size=num_complex_pairs)
    complex_roots = radii * np.exp(1j * thetas)
    roots = np.concatenate([complex_roots, np.conjugate(complex_roots)])

    # Si p es impar, se necesita una raíz real adicional entre -1 y 1.
    if p % 2 != 0:
        real_root = np.random.uniform(-1, 1, size=1)
        roots = np.concatenate([roots, real_root])
        
    # Construye el polinomio a partir de las raíces.
    poly_coeffs = polyfromroots(roots)
    
    # Convierte los coeficientes del polinomio a los coeficientes phi del proceso AR.
    phi = -np.real(poly_coeffs[:-1][::-1])
    return phi.reshape(1, p)

def generate_stationary_var(n: int, p: int, eig_max: float = 0.98):
    """
    Genera los parámetros para un modelo VAR(p) aleatorio y garantizado 
    como estacionario.

    Argumentos:
        n (int): El número de variables del sistema (inp_dim).
        p (int): El orden de rezagos del VAR.
        eig_max (float): El módulo máximo para los eigenvalores generados (debe ser < 1).
    """
    if eig_max >= 1.0:
        raise ValueError("eig_max debe ser menor que 1 para garantizar la estacionariedad.")
        
    n_eigs = n * p

    # Generar Eigenvalores Estables (dentro del círculo unitario)
    eigs = []
    for _ in range(n_eigs // 2):
        modulus = np.random.uniform(0, eig_max)
        angle = np.random.uniform(0, np.pi)
        eig = modulus * np.exp(1j * angle)
        eigs.extend([eig, np.conj(eig)]) # Añadir par conjugado
        
    if n_eigs % 2 != 0:
        eigs.append(np.random.uniform(-eig_max, eig_max))
    
    eigs = np.array(eigs)

    # Construir la Matriz Compañera F
    D = np.diag(eigs)
    P = np.random.randn(n_eigs, n_eigs)
    F = P @ D @ np.linalg.inv(P)
    F = F.real


    # Extraer las Matrices de Coeficientes Phi de F
    phi_stacked = F[:n, :]
    phi = np.array(np.split(phi_stacked, p, axis=1))

    # Generar Intercepto C y Matriz de Covarianzas Omega
    c = np.random.randn(n) * 0.1
    rand_matrix = np.random.randn(n, n) * 0.5
    omega = rand_matrix.T @ rand_matrix + np.eye(n) * 0.1 # Asegura que no sea singular
    
    return phi, c, omega

def initialize_params(params_distribution, **kwargs):
    """
    Inicializa los parámetros de un modelo de forma flexible.

    Puede recibir los parámetros directamente, generarlos para asegurar estacionariedad,
    o crearlos a partir de una distribución de probabilidad específica.
    """
    # Si no se especifica distribución, no hay parámetros.
    if params_distribution is None:
        return None

    # Si se pasan los parámetros directamente como una lista o array.
    if isinstance(params_distribution, (list, np.ndarray)):
        return np.array(params_distribution)

    # Si se solicita generar parámetros que garanticen estacionariedad.
    if params_distribution == 'stationary':
        return generate_stationary_phi(kwargs.get('p', 2)).flatten()

    # Busca la distribución solicitada en el diccionario de generadores.
    generator_func = DISTRIBUTION_GENERATORS.get(params_distribution)

    if generator_func:
        try:
            # Llama a la función generadora correspondiente.
            return generator_func(**kwargs)
        except KeyError as e:
            raise TypeError(f"Falta el argumento requerido: {e} para la distribución '{params_distribution}'")
    else:
        raise ValueError(f"'{params_distribution}' no es un valor válido para params_distribution.")
    
def create_time_series(kind, n_obs, randseed=None, **kwargs):
    """
    Crea series de tiempo con propiedades específicas.
    
    Args:
        kind (str): Tipo de serie ('stationary', 'unit_root', 'break', 'outlier', 'nonlinear_trend').
        n_obs (int): Número de observaciones.
        kwargs: Parámetros adicionales como break_point, outlier_magnitude, etc.

    Returns:
        np.ndarray: La serie de tiempo simulada.
    """
    

    if kind == 'stationary':
        if randseed is not None: np.random.seed(randseed)
        # Serie estacionaria (proceso AR(1))
        # Phi = 0.5 (abs(phi) < 1)
        ar_params = np.array([0.5])
        ma_params = np.array([0])
        return sm.tsa.arma_generate_sample(ar=np.r_[1, -ar_params], ma=np.r_[1, ma_params], nsample=n_obs)
    
    elif kind == 'unit_root':
        if randseed is not None: np.random.seed(randseed+2)
        # Proceso de paseo aleatorio (random walk)
        series = np.cumsum(np.random.normal(0, 1, n_obs))
        return series
    
    elif kind == 'break':
        if randseed is not None: np.random.seed(randseed+4)
        # Paseo aleatorio con quiebre estructural
        break_point = kwargs.get('break_point', int(n_obs * 0.5))
        break_magnitude = kwargs.get('break_magnitude', 5.0)
        series = np.cumsum(np.random.normal(0, 1, n_obs))
        series[break_point:] += break_magnitude
        return series
    
    elif kind == 'outlier':
        if randseed is not None: np.random.seed(randseed+6)
        # Paseo aleatorio con un valor atípico aditivo
        outlier_point = kwargs.get('outlier_point', int(n_obs * 0.7))
        outlier_magnitude = kwargs.get('outlier_magnitude', 10.0)
        series = np.cumsum(np.random.normal(0, 1, n_obs))
        series[outlier_point] += outlier_magnitude
        return series
    
    elif kind == 'nonlinear_trend':
        if randseed is not None: np.random.seed(randseed+8)
        # Paseo aleatorio con una tendencia no lineal
        series = np.cumsum(np.random.normal(0, 1, n_obs))
        trend = np.linspace(0, 1, n_obs)
        series += kwargs.get('trend_magnitude', 5.0) * trend**2
        return series
    
    else:
        raise ValueError("Tipo de serie no reconocido.")
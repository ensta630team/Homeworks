import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
import statsmodels.api as sm

################################################################################
################################################################################

# Determinar los Ψ_i_s por recurrencia
def compute_psi_sequence(Phi, p, n_terms):
    """
    Calcula la sucesión Ψ_i para i = 0, 1, 2, ..., n_terms-1
    """
    if len(Phi) != p:
        raise ValueError(f"Se esperaban {p} matrices Φ, pero se recibieron {len(Phi)}")
    
    n = Phi[0].shape[0]
    Psi = [np.eye(n)]  # Ψ₀ = I
    
    for s in range(1, n_terms):
        suma = np.zeros((n, n))
        for i in range(1, p+1):
            if s - i >= 0: 
                suma += np.dot(Phi[i-1], Psi[s-i])
        Psi.append(suma)
    
    return Psi

################################################################################
################################################################################

# Cálculo del IRF mediante descomposición de Cholesky
def compute_IRF_VAR(Psi, omega, s):
    """
    Cálculo de IRF usando descomposición de Cholesky
    """
    L = cholesky(omega, lower=True)
    IRF_s = np.dot(Psi[s], L)
    return IRF_s

################################################################################
################################################################################

# Descomposición de la Varianza
def compute_variance_decomposition(Psi, omega, horizon):
    """
    Calcula la descomposición de varianza del error de pronóstico
    
    Argumentos:
        Psi: Lista de matrices Ψ
        omega: Matriz de covarianza
        horizon: Horizonte temporal
    
    Retorna:
        Matriz n x n donde cada fila suma 1 (porcentajes)
    """
    n = omega.shape[0]
    L = cholesky(omega, lower=True)
    
    # Calcular contribución de cada shock a la varianza
    fevd = np.zeros((n, n))
    total_variance = np.zeros(n)
    
    for s in range(horizon + 1):
        theta_s = np.dot(Psi[s], L)
        for i in range(n):
            for j in range(n):
                fevd[i, j] += theta_s[i, j]**2
    
    # Normalizar a porcentajes
    row_sums = fevd.sum(axis=1, keepdims=True)
    fevd_percent = fevd / row_sums * 100
    
    return fevd_percent


################################################################################
################################################################################

# Intervalos de Confianza Bootstraps
def bootstrap_IRF_intervals(X, p, n_terms, n_boot=1000, alpha=0.05):
    """
    Bootstrap completo que re-estima el VAR en cada réplica
    """
    T, n = X.shape
    IRF_boot = []
    
    # Estimar modelo original para obtener coeficientes iniciales
    model = sm.tsa.VAR(X)
    results = model.fit(p)
    Phi_original = [results.coefs[i] for i in range(p)]
    omega_original = results.sigma_u
    residuals_original = results.resid
    
    for i in range(n_boot):
        try:
            # Generar residuos bootstrap (resampling con reemplazo)
            indices = np.random.choice(T-p, size=T-p, replace=True)
            residuals_boot = residuals_original[indices]
            
            # Generar datos bootstrap recursivamente
            X_boot = np.zeros_like(X)
            X_boot[:p] = X[:p]  # Valores iniciales
            
            for t in range(p, T):
                # Componente determinística (constantes, etc.)
                deterministic = results.intercept if hasattr(results, 'intercept') else 0
                
                # Componente autorregresiva
                ar_component = 0
                for lag in range(1, p+1):
                    ar_component += np.dot(Phi_original[lag-1], X_boot[t-lag])
                
                X_boot[t] = deterministic + ar_component + residuals_boot[t-p]
            
            # Re-estimar VAR con datos bootstrap
            model_boot = sm.tsa.VAR(X_boot)
            results_boot = model_boot.fit(p)
            Phi_boot = [results_boot.coefs[lag] for lag in range(p)]
            omega_boot = results_boot.sigma_u
            
            # Calcular IRF
            Psi_boot = compute_psi_sequence(Phi_boot, p, n_terms)
            IRF_sequence = [compute_IRF_VAR(Psi_boot, omega_boot, s) for s in range(n_terms)]
            IRF_boot.append(IRF_sequence)
            
        except:
            continue
    
    # Calcular intervalos de confianza
    IRF_boot = np.array(IRF_boot)
    lower_percentile = 100 * alpha/2
    upper_percentile = 100 * (1 - alpha/2)
    
    IRF_lower = np.percentile(IRF_boot, lower_percentile, axis=0)
    IRF_upper = np.percentile(IRF_boot, upper_percentile, axis=0)
    
    return IRF_lower, IRF_upper
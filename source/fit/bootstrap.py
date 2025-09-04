import numpy as np
import functools
import pandas as pd
from tqdm import tqdm
import multiprocess
import os

from source.models.var import VAR

# ==============================================================================
# DECORADOR PARALELO
# ==============================================================================
def _parallel_runner(n_iterations: int = 1000, n_jobs: int = -1):
    """
    Decorador que ejecuta una función en paralelo y devuelve una lista de resultados.
    A diferencia de monte_carlo_mp, NO convierte los resultados a un DataFrame,
    permitiendo manejar salidas multidimensionales como las IRF.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if n_jobs == -1:
                num_processes = os.cpu_count()
            else:
                num_processes = min(n_jobs, os.cpu_count())

            print(f"Iniciando simulación paralela con {n_iterations} iteraciones en {num_processes} procesos...")
            
            # Crea una función de trabajo con los argumentos fijos
            worker_func = functools.partial(func, *args, **kwargs)
            
            with multiprocess.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(worker_func, range(n_iterations)), 
                                    total=n_iterations, 
                                    desc="Bootstrap en Paralelo"))
            # Devuelve la lista de resultados directamente
            return results
        return wrapper
    return decorator

# ==============================================================================
# FUNCIÓN DE UN SOLO PASO
# ==============================================================================
def _step_bootstrap(
    _: int, # Argumento requerido por el decorador, no se usa
    fitted_var: VAR, 
    statistic_fn
):
    """
    Realiza UNA única replicación del procedimiento de bootstrap.
    """
    # ... (El resto del código de esta función es idéntico al de la respuesta anterior)
    T, n_vars = fitted_var.y.shape
    p = fitted_var.p
    residuals = fitted_var.residuals
    T_res = residuals.shape[0]
    phi_pred_matrix = fitted_var.phi.transpose(1, 0, 2).reshape(n_vars, p * n_vars)
    resampled_indices = np.random.randint(0, T_res, size=T_res)
    resampled_residuals = residuals[resampled_indices]
    y_boot = np.zeros_like(fitted_var.y)
    y_boot[:p, :] = fitted_var.y[:p, :]
    for t in range(p, T):
        y_lags_flat = y_boot[t-p:t, :][::-1].flatten()
        y_boot[t, :] = fitted_var.c + phi_pred_matrix @ y_lags_flat + resampled_residuals[t-p]
    
    try:
        var_boot = VAR(inp_dim=n_vars, p=p)
        var_boot.fit(y_boot)
        return statistic_fn(var_boot)
    except Exception:
        return None

# ==============================================================================
# FUNCIÓN ORQUESTADORA
# ==============================================================================
def run_bootstrap_mp(
    fitted_var: VAR,
    statistic_fn,
    n_replications: int = 1000,
    confidence_level: float = 0.95,
    n_jobs: int = 1
):
    """
    Ejecuta el bootstrap en paralelo y maneja correctamente los resultados multidimensionales.
    """
    point_estimate = statistic_fn(fitted_var)

    # Decoramos nuestra función de un solo paso con el nuevo runner
    @_parallel_runner(n_iterations=n_replications, n_jobs=n_jobs)
    def decorated_runner(_, fitted_var, statistic_fn):
        return _step_bootstrap(_, fitted_var, statistic_fn)

    # Ejecutamos la simulación
    bootstrap_results = decorated_runner(fitted_var=fitted_var, statistic_fn=statistic_fn)
    
    # Filtramos los 'None' si alguna réplica falló
    valid_results = [res for res in bootstrap_results if res is not None]
    results_array = np.array(valid_results)
    
    if results_array.ndim < 2:
        print("Error: No se pudieron obtener resultados válidos del bootstrap.")
        return point_estimate, None, None

    print(f"\nCálculo finalizado. {len(valid_results)}/{n_replications} réplicas exitosas.")

    # Calculamos percentiles directamente sobre el array de NumPy
    alpha = (1 - confidence_level) / 2
    lower_bound = np.percentile(results_array, alpha * 100, axis=0)
    upper_bound = np.percentile(results_array, (1 - alpha) * 100, axis=0)

    return point_estimate, lower_bound, upper_bound
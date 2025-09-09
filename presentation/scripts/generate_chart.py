# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from source.data.sampling import create_time_series
# from source.fit import inference as testmodule
# from source.display.hw import hw3 as plot_utils
# from source.display.hw.hw1 import save_figure

# NITER = 50
# nobs = 200
# # Hacer 1000 Iteraciones 
# results =  []
# for i in range(NITER):
#     # En cada iteracion simulamos un proceso estocastico y_t = y_{t-1} + u_t
#     dataset = create_time_series('unit_root', n_obs=nobs)
    
#     # Evaluamos estacionaridad en cada uno de los modelos utilizando la serie

#     metrics_0 = testmodule.test_df(dataset)
#     metrics_1 = testmodule.test_adf(dataset, max_k=10, model_type='c', alpha=0.05)
#     metrics_2 = testmodule.test_pp(dataset)
#     metrics_3 = testmodule.test_kpss(dataset)
#     metrics_4 = testmodule.test_perron(dataset, varname='unit_root', breakpoint=50)
#     metrics_5 = testmodule.test_zivot_andrews_v2(dataset)
#     metrics_6 = testmodule.test_bierens(dataset, max_k=5, max_m=3, alpha=0.05)
#     metrics_7 = testmodule.test_murray_nelson(dataset, periodo=[10, 15], k=5)
#     metrics_8 = testmodule.test_gls(dataset, p=13)

#     # Guardamos el p-value y el rechazo
#     columns = []
#     for j, m in enumerate([metrics_0, metrics_1, metrics_2, metrics_3, metrics_4, metrics_5, metrics_6, metrics_7, metrics_8]):
#         if i == 0:
#             m = m.rename(columns={'p-value': str(i)})
#             columns.append(m[['model', '0']])
#         else:
#             m = m.rename(columns={'p-value': str(i)})
#             columns.append(m[str(i)])

#     columns = pd.concat(columns, axis=0, ignore_index=True)
    
#     results.append(columns)

# # Agregar una columna que tenga 1 en caso de ser rechazado al menos una vez 
# foo =  pd.concat(results, axis=1)
# foo.to_csv('./presentation/backup/bigtable.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Asume que estas importaciones están disponibles
from source.data.sampling import create_time_series
from source.fit import inference as testmodule

# --- FUNCIÓN PARA UNA ÚNICA ITERACIÓN DE LA SIMULACIÓN ---
def run_single_iteration(n_observations: int) -> pd.Series:
    """
    Ejecuta una única iteración de la simulación de Monte Carlo.
    
    Args:
        n_observations (int): Número de observaciones para la serie de tiempo.
        
    Returns:
        pd.Series: Una Serie con el p-valor de cada test, indexada por el nombre del modelo.
    """
    # 1. Simular un proceso estocástico con raíz unitaria
    dataset = create_time_series('unit_root', n_obs=n_observations)
    
    # 2. Evaluar la estacionaridad con todos los modelos
    # Cada función testmodule.test_* retorna un DataFrame. 
    # Usamos .iloc[0] para acceder a la primera fila, donde están los resultados,
    # y luego ['p-value'] para obtener el valor del p-value.
    
    p_values = {
        'DF': testmodule.test_df(dataset).min()['p-value'],
        'ADF': testmodule.test_adf(dataset, max_k=10, model_type='c', alpha=0.05).min()['p-value'],
        'PP': testmodule.test_pp(dataset).min()['p-value'],
        'KPSS': testmodule.test_kpss(dataset).min()['p-value'],
        'Perron': testmodule.test_perron(dataset, varname='unit_root', breakpoint=50).min()['p-value'],
        'Zivot-Andrews': testmodule.test_zivot_andrews_v2(dataset).min()['p-value'],
        'Bierens': testmodule.test_bierens(dataset, max_k=5, max_m=3, alpha=0.05).min()['p-value'],
        'Murray-Nelson': testmodule.test_murray_nelson(dataset, periodo=[10, 15], k=5).min()['p-value'],
        'GLS': testmodule.test_gls(dataset, p=13).min()['p-value']
    }
    
    # 3. Retornar los resultados como una Serie de Pandas
    return pd.Series(p_values)

# --- LÓGICA PRINCIPAL (EJECUTADA POR EL SCRIPT) ---
if __name__ == '__main__':
    # Configuración de la simulación
    NITER = 1000
    nobs = 200

    print(f"Iniciando simulación con {NITER} iteraciones en {mp.cpu_count()} núcleos...")
    
    # Crear un pool de procesos. El número de procesos se determina automáticamente.
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Usar pool.map para distribuir las tareas.
        # Esto es la "magia" del paralelismo.
        results = pool.map(run_single_iteration, [nobs] * NITER)

    # Concatenar todos los resultados en un solo DataFrame
    final_df = pd.concat(results, axis=1)
    final_df.columns = [f'Iteración_{i}' for i in range(NITER)]
    final_df = final_df.T
    
    # Guardar los resultados en un archivo CSV
    output_path = './presentation/backup/bigtable_multiprocess.csv'
    final_df.to_csv(output_path, index=True)
    print(f"Simulación completa. Resultados guardados en '{output_path}'")
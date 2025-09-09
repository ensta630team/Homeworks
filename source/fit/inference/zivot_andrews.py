import pandas as pd
import numpy as np 
import os
import multiprocessing
import statsmodels.api as sm

from source.data.loaders import load_json
from source.fit.inference.perron import create_model_a, create_model_b, create_model_c
from source.fit.model_selection import find_best_p


from statsmodels.tsa.stattools import zivot_andrews

import pandas as pd
import numpy as np 
import os
import multiprocessing
import statsmodels.api as sm
from statsmodels.tsa.stattools import zivot_andrews

from source.fit.inference.perron import create_model_a, create_model_b, create_model_c
from source.fit.model_selection import find_best_p


# ===========================================================================
# FUNCIONES CORREGIDAS PARA EL MANEJO DE CLAVES DEL JSON
# ===========================================================================
def _clean_key_to_float(key):
    """Limpia una clave de cadena y la convierte a float."""
    key_str = str(key).replace('%', '')
    if key_str == 'inf':
        return np.inf
    return float(key_str)

def get_zivotandrews_p_value(t_statistic, breakpoint, n_obs, model_type="model_a", 
                             json_path='./data/zivotandrews_critical_values.json'):
    """
    Estima el p-valor para un estadístico del test de ZA dado,
    usando interpolación lineal sobre los cuantiles para un breakpoint específico.
    """
    critical_values_data = load_json(json_path)
    
    # Obtenemos la tabla de valores críticos para el modelo, excluyendo la tabla 'inf'
    critical_values_table = {k:v for k,v in critical_values_data[model_type].items() if k != "inf"}

    # Convertir el breakpoint entero a su valor lambda (proporción)
    lam = breakpoint / n_obs
    
    # Encontramos los dos breakpoints de la tabla entre los que se encuentra nuestro lambda
    # Tomamos la lista de lambdas de cualquier nivel de significancia, por ejemplo, "1.0%"
    any_sig_key = next(iter(critical_values_table.keys()))
    table_lambdas_keys = critical_values_table[any_sig_key].keys()
    table_lambdas = sorted([_clean_key_to_float(k) for k in table_lambdas_keys])
    
    try:
        lambda_1_val = max([l for l in table_lambdas if l <= lam])
        lambda_2_val = min([l for l in table_lambdas if l >= lam])
    except ValueError:
        if lam < min(table_lambdas):
            lambda_1_val = lambda_2_val = min(table_lambdas)
        else:
            lambda_1_val = lambda_2_val = max(table_lambdas)

    lambda_1_key = [k for k in table_lambdas_keys if _clean_key_to_float(k) == lambda_1_val][0]
    lambda_2_key = [k for k in table_lambdas_keys if _clean_key_to_float(k) == lambda_2_val][0]
    
    cv_quantile_pairs = []
    
    for sig_key, breakpoints in critical_values_table.items():
        sig_float = _clean_key_to_float(sig_key)
        quantile = sig_float / 100
        
        cv_1 = breakpoints.get(lambda_1_key)
        if cv_1 is None: continue

        if lambda_1_val == lambda_2_val:
            interpolated_cv = cv_1
        else:
            cv_2 = breakpoints.get(lambda_2_key)
            if cv_2 is None: continue
            
            interpolated_cv = cv_1 + (lam - lambda_1_val) * (cv_2 - cv_1) / (lambda_2_val - lambda_1_val)

        cv_quantile_pairs.append((interpolated_cv, quantile))

    sorted_pairs = sorted(cv_quantile_pairs)
    sorted_cvs = [pair[0] for pair in sorted_pairs]
    sorted_quantiles = [pair[1] for pair in sorted_pairs]
    
    p_value = np.interp(t_statistic, sorted_cvs, sorted_quantiles)
    
    if t_statistic < sorted_cvs[0]:
        return sorted_quantiles[0]
    if t_statistic > sorted_cvs[-1]:
        return sorted_quantiles[-1]
        
    return p_value

def _process_breakpoint(args):
    """
    Función 'worker' que procesa un único breakpoint.
    Está diseñada para ser llamada por un pool de multiprocesamiento.
    """
    # Desempaquetamos los argumentos
    bp, serie, model_type, input_fn, variablename = args
    try:
        # Crear el input dle modelo
        presult = find_best_p(serie, bp, input_fn=input_fn, p_max=6)
        
        if presult is None or presult.empty:
            return None
            
        p_star = presult.loc[presult['t'].idxmin()]['p'].astype(int)
        
        # Guardar para ver despues
        backuppath = os.path.join('./presentation/backup/perron/', model_type, variablename, f'bp_{bp}.csv')
        if not os.path.exists(os.path.dirname(backuppath)):
            os.makedirs(os.path.dirname(backuppath))
        presult.to_csv(backuppath)

        X, y = input_fn(serie=serie, break_point=bp, p=p_star)

        model = sm.OLS(y, X)
        results = model.fit()
        tstat = results.tvalues[3]
        
        pvalue = get_zivotandrews_p_value(tstat, breakpoint=bp, n_obs=len(serie), model_type=model_type)

        return pd.DataFrame({
            'model': [model_type],
            'break': [bp],
            'pstar': [p_star],
            'statistic': [tstat],
            'p-value': [pvalue]
        })

    except Exception as e:
        print(f"Error en breakpoint {bp} para el modelo {model_type}: {e}")
        return None

def grid_zivot_parallel(serie, trim=0.15, variablename=''):
    n_obs = len(serie)
    grid = np.arange(int(n_obs * trim), int(n_obs * (1. - trim)), 1)
    
    all_model_results = []
    for model_type, input_fn in [('model_a', create_model_a),
                                 ('model_b', create_model_b),
                                 ('model_c', create_model_c)]:
        tasks = [(bp, serie, model_type, input_fn, variablename) for bp in grid]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
            results_list = pool.map(_process_breakpoint, tasks)

        valid_results = [res for res in results_list if res is not None]

        if valid_results:
            all_model_results.append(pd.concat(valid_results, axis=0))

    if not all_model_results:
        return pd.DataFrame()

    return pd.concat(all_model_results, axis=0)
    	
def test_zivot_andrews(serie, varname=''):
    print('Test Zivot-Andrews')
    dataframes = grid_zivot_parallel(serie=serie, variablename=varname)

    if dataframes.empty:
        print("Advertencia: La búsqueda en grilla no produjo resultados.")
        return pd.DataFrame()
        
    sorted_df = dataframes.sort_values('statistic', ascending=True)
    result = sorted_df.drop_duplicates(subset='model', keep='first')
    return result.sort_values('model')



def test_zivot_andrews_v2(serie, varname=''):
    """
    Realiza el test de raíz unitaria de Zivot-Andrews con la función de statsmodels.
    Compara los tres modelos (A, B, C) y retorna el mejor resultado para cada uno.
    """
    print('Test Zivot-Andrews con statsmodels')
    
    results = []

    # Mapeo de tus modelos a los parámetros de regresión de statsmodels
    model_types = {
        'model_a': 'c',   # Quiebre en el nivel (intercept)
        'model_b': 't',   # Quiebre en la tendencia
        'model_c': 'ct'   # Quiebre en el nivel y la tendencia
    }
    
    # Iterar sobre cada tipo de modelo para obtener el mejor resultado de cada uno
    for model_name, regression_type in model_types.items():
        try:
            # La función de statsmodels encuentra el mejor breakpoint y el p-value
            za_result = zivot_andrews(x=serie, 
                                      regression=regression_type, 
                                      autolag='AIC')
            
            # Extraer los valores con los índices correctos
            tstat = za_result[0]
            pvalue = za_result[1]
            pstar = za_result[3] # Número de rezagos óptimo (baselag)
            breakpoint_idx = za_result[4] # Índice del breakpoint óptimo (bpidx)

            results.append({
                'model': model_name,
                'break': breakpoint_idx,
                'pstar': pstar,
                'statistic': tstat,
                'p-value': pvalue
            })
        except Exception as e:
            print(f"Advertencia: No se pudo ejecutar el test para el modelo {model_name}. Error: {e}")
            results.append({
                'model': model_name,
                'break': None,
                'pstar': None,
                'statistic': None,
                'p-value': None
            })

    # Convertir la lista de resultados en un DataFrame y ordenar
    if not results:
        return pd.DataFrame()
        
    df_results = pd.DataFrame(results)
    
    return df_results.sort_values('model')
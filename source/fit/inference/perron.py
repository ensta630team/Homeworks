import statsmodels.api as sm
import multiprocessing
import pandas as pd
import numpy as np 
import os

from source.data.transform import create_lagged_dataset
from source.data.sampling import create_time_series
from source.data.loaders import load_json
from source.models.ols import OLS
from source.fit.error import get_standard_error
from source.fit.model_selection import hannan_rissanen, find_best_p


def get_perron_critical_value(lambda_val, model_type="model_a", significance="5%", json_path='./data/perron_critical_values.json'):
    """
    Obtiene el valor crítico de Perron, interpolando si es necesario.

    Args:
        lambda_val (float): La posición del quiebre (e.g., 0.25).
        model_type (str): 'model_a', 'model_b', o 'model_c'.
        significance (str): '1%', '2.5%', '5%', o '10%'.
        json_path (str): Ruta al archivo JSON con los valores críticos.

    Returns:
        float: El valor crítico (interpolado si es necesario).
    """
    critical_values_data = load_json(json_path)

    table = critical_values_data[model_type][significance]
    
    # Convertir las claves (lambdas de la tabla) a float
    table_lambdas = sorted([float(k) for k in table.keys()])
    
    # Caso 1: El lambda está exactamente en la tabla
    if lambda_val in table_lambdas:
        return table[str(lambda_val)]

    # Caso 2: Necesitamos interpolar
    # Encontrar los lambdas que rodean nuestro valor
    lambda_1 = max([l for l in table_lambdas if l < lambda_val])
    lambda_2 = min([l for l in table_lambdas if l > lambda_val])

    cv_1 = table[str(lambda_1)]
    cv_2 = table[str(lambda_2)]

    # Aplicar la fórmula de interpolación lineal
    interpolated_cv = cv_1 + (lambda_val - lambda_1) * (cv_2 - cv_1) / (lambda_2 - lambda_1)
    
    return interpolated_cv

def _create_base_regression(serie, break_point, p=1):
    """
    Función auxiliar robusta que evita eliminar dummies por error.
    """
    df = pd.DataFrame({'y': serie})
    n_total = len(df)

    # Crear TODAS las variables que puedan tener NaNs al principio
    df['delta_y'] = df['y'].diff()
    df['y_lag1'] = df['y'].shift(1)
    for i in range(1, p + 1):
        df[f'delta_y_lag{i}'] = df['delta_y'].shift(i)

    # Ahora que todos los NaNs están presentes, IDENTIFICAR el primer índice válido
    first_valid_index = df.dropna().index[0]

    # Crear las dummies y variables determinísticas SOLAMENTE en el rango válido
    df['intercept'] = 1
    df['trend'] = np.arange(1, n_total + 1)
    
    # =====================================================================
    # Dummies =============================================================
    # =====================================================================
    # Shock transitorio
    df['d_tb'] = 0
    if break_point + 1 >= first_valid_index:
        df.loc[break_point + 1, 'd_tb'] = 1
    
    # Shock permanente (cambio de nivel)
    df['d_u'] = 0
    df.loc[max(break_point + 1, first_valid_index):, 'd_u'] = 1
    
    # Changing growth 
    df['dt_star'] = 0
    # Creamos la serie para el cambio de pendiente
    trend_shift_values = df.loc[max(break_point + 1, first_valid_index):, 'trend'] - (break_point + 1)
    df.loc[max(break_point + 1, first_valid_index):, 'dt_star'] = trend_shift_values
    
    # 4. Finalmente, eliminar todas las filas con cualquier NaN
    df.dropna(inplace=True)
    
    return df

# ===========================================================================
# Modelos ===================================================================
# ===========================================================================
def create_model_a(serie, break_point, p=1):
    """Crash Model"""
    df = _create_base_regression(serie, break_point, p)
    
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'd_u', 'd_tb']
    for i in range(1, p + 1):
        regressor_cols.append(f'delta_y_lag{i}')
    df.dropna(inplace=True)

    X = df[regressor_cols].values
    y = df['delta_y'].values
    return X, y

def create_model_b(serie, break_point, p=1):
    """Prepara los datos para el Modelo B usando la función base."""
    df = _create_base_regression(serie, break_point, p)

    # Definir regresores y alinear
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'dt_star', 'd_u']
    for i in range(1, p + 1):
        regressor_cols.append(f'delta_y_lag{i}')
    df.dropna(inplace=True)
    
    X = df[regressor_cols].values
    y = df['delta_y'].values
    return X, y

def create_model_c(serie, break_point, p=1):
    """
    Prepara los datos para el Modelo C reutilizando la lógica base.
    """
    df = _create_base_regression(serie, break_point, p)

    # 3. Definir la lista completa de regresores
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'd_u', 'd_tb','dt_star']
    for i in range(1, p + 1):
        regressor_cols.append(f'delta_y_lag{i}')
    df.dropna(inplace=True)

    X = df[regressor_cols].values
    y = df['delta_y'].values
    return X, y

def get_perron_p_value(t_statistic, lambda_val, 
                       model_type="model_a", 
                       json_path='./data/perron_critical_values.json'):
    """
    Estima el p-valor para un estadístico del test de Perron dado,
    usando interpolación lineal sobre los cuantiles conocidos.
    """
    critical_values_data = load_json(json_path)
    
    # 1. RECOLECTAR TODOS LOS PUNTOS CONOCIDOS (CV, Quantile)
    quantiles = []
    cvs = []
    significance_levels = critical_values_data[model_type].keys()
    
    for sig in significance_levels:
        quantile = float(sig.replace('%', '')) / 100
        quantiles.append(quantile)
        
        cv = get_perron_critical_value(lambda_val, model_type, sig, json_path)
        cvs.append(cv)
        
    # Ordenar los puntos para la interpolación
    sorted_pairs = sorted(zip(cvs, quantiles))
    sorted_cvs, sorted_quantiles = zip(*sorted_pairs)

    p_value = np.interp(t_statistic, sorted_cvs, sorted_quantiles)

    return p_value

def grid_perron_parallel(serie, trim=0.15, variablename=''):
    """
    Versión paralela de grid_perron que distribuye la búsqueda
    de breakpoints entre todos los núcleos de la CPU disponibles.
    """
    n_obs = len(serie)
    grid = np.arange(int(n_obs * trim), int(n_obs * (1. - trim)), 1)
    
    all_model_results = []
    for model_type, input_fn in [('model_a', create_model_a),
                                 ('model_b', create_model_b),
                                 ('model_c', create_model_c)]:
        

        # 1. Prepara la lista de tareas (una tupla de args para cada bp)
        tasks = [(bp, serie, model_type, input_fn, variablename) for bp in grid]
        
        # 2. Crea un pool de procesos y distribuye las tareas
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
            results_list = pool.map(_process_breakpoint, tasks)

        # 3. Filtra los resultados que pudieron haber fallado (son None)
        valid_results = [res for res in results_list if res is not None]

        if valid_results:
            all_model_results.append(pd.concat(valid_results, axis=0))

    if not all_model_results:
        return pd.DataFrame()

    return pd.concat(all_model_results, axis=0)
    	
def test_perron(serie, varname=''):
    print('Test Perron')

    # Si el breakpoint es desconocido, elegimos la función a ejecutar
    dataframes = grid_perron_parallel(serie=serie, variablename=varname)

    # El resto de la lógica para encontrar el mejor resultado no cambia
    if dataframes.empty:
        print("Advertencia: La búsqueda en grilla no produjo resultados.")
        return pd.DataFrame()
        
    sorted_df = dataframes.sort_values('statistic', ascending=True)
    result = sorted_df.drop_duplicates(subset='model', keep='first')
    return result.sort_values('model')

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
        # ols_model = OLS()
        # ols_model.fit(X, y)
        # beta_hat = ols_model.beta

        # y_pred = ols_model.predict(X)
        # s_hat = get_standard_error(X, y, y_pred=y_pred)
        # tstat = beta_hat.flatten()[3] / s_hat[3] 

        lambda_val = bp / len(serie)
        pvalue = get_perron_p_value(tstat, lambda_val, model_type=model_type)

        return pd.DataFrame({
            'model': [model_type],
            'break': [bp],
            'pstar': [p_star],
            'statistic': [tstat],
            'p-value': [pvalue]
        })

    except Exception as e:
        # Si algo falla para un breakpoint, devolvemos None para ignorarlo
        return None
    

import numpy as np 
import pandas as pd
import statsmodels.api as sm

from source.data.transform import create_lagged_dataset
from source.data.sampling import create_time_series
from source.data.loaders import load_json
from source.models.ols import OLS
from source.fit.error import get_standard_error
from source.fit.model_selection import hannan_rissanen



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


def _create_base_regression_df(serie, break_point, p=1):
    """
    Función auxiliar para crear un DataFrame con todas las variables
    comunes a los tres modelos de Perron.
    """
    df = pd.DataFrame({'y': serie})
    n_total = len(df)

    # Variables base
    df['delta_y'] = df['y'].diff()
    df['y_lag1'] = df['y'].shift(1)
    
    # Variables determinísticas
    df['intercept'] = 1
    df['trend'] = np.arange(1, n_total + 1)
    
    # Dummy Pulso
    df['d_tb'] = 0
    if break_point + 1 < n_total:
        df.loc[break_point + 1, 'd_tb'] = 1
    
    # Dummy cambio de nivel
    df['du'] = 0
    df.loc[break_point + 1:, 'du'] = 1
    
    # Rezagos de la variable diferenciada
    for i in range(1, p + 1):
        df[f'delta_y_lag{i}'] = df['delta_y'].shift(i)
        
    return df

def create_model_a(serie, break_point, p=1):
    """Prepara los datos para el Modelo A usando la función base."""
    df = _create_base_regression_df(serie, break_point, p)
    
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'd_tb']
    for i in range(1, p + 1):
        regressor_cols.append(f'delta_y_lag{i}')
    df.dropna(inplace=True)

    X = df[regressor_cols].values
    y = df['delta_y'].values
    return X, y

def create_model_b(serie, break_point, p=1):
    """Prepara los datos para el Modelo B usando la función base."""
    df = _create_base_regression_df(serie, break_point, p)

    # Definir regresores y alinear
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'du']
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
    df = _create_base_regression_df(serie, break_point, p)

    # 3. Definir la lista completa de regresores
    regressor_cols = ['intercept', 'y_lag1', 'trend', 'd_tb', 'du']
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
        
        # Obtiene el valor crítico para nuestro lambda específico
        cv = get_perron_critical_value(lambda_val, model_type, sig, json_path)
        cvs.append(cv)
        
    # Ordenar los puntos para la interpolación
    sorted_pairs = sorted(zip(cvs, quantiles))
    sorted_cvs, sorted_quantiles = zip(*sorted_pairs)
    
    # 2. REALIZAR LA INTERPOLACIÓN LINEAL
    # np.interp(x_nuevo, x_conocidos, y_conocidos)
    p_value = np.interp(t_statistic, sorted_cvs, sorted_quantiles)

    return p_value

def test_perron(serie, break_point=None):
    print('Test Perron')

    if break_point is None:
        break_point = int(len(serie)*0.5)

    dataframes = []
    for model_type, input_fn in [('model_a', create_model_a), ('model_b', create_model_b), ('model_c', create_model_c)]:
            
        table = hannan_rissanen(serie, max_rezagos=30, verbose=False, model_type='ar', inp_fn=input_fn)
        table.sort_values('BIC', inplace=True) 
        p_star = table.index[0]

        X, y = input_fn(serie=serie, break_point=break_point, p=p_star)
        
        ols_model = OLS()
        _ = ols_model.fit(X, y)
        beta_hat = ols_model.beta

        y_pred = ols_model.predict(X)
        s_hat = get_standard_error(X, y, y_pred=y_pred)
        tstat = beta_hat.flatten() / s_hat
        tstat = tstat[1] # param asociado a la hipotesis

        lambda_val = break_point / len(serie)
        pvalue = get_perron_p_value(tstat, lambda_val, model_type=model_type)

        results = pd.DataFrame()
        results['model'] = [model_type]
        results['statistic'] = [tstat]
        results['p-value'] = [pvalue]
        dataframes.append(results)

    return pd.concat(dataframes, axis=0)
    	

 	


if __name__ == '__main__':

    types_of_series = ['stationary', 'unit_root', 'break', 'outlier', 'nonlinear_trend']
    nobs = 500
    break_point = 100
    dataset = {}
    for kind in types_of_series:
        dataset[kind] = create_time_series(kind, nobs, randseed=42, break_point=break_point) # para que todos los test se comparen con los mismos datos
    dataset['t'] = np.linspace(0,  1, nobs)


    # Y para ejecutar el test
    print("\n--- Testing 'break' series ---")
    metrics = test_perron(dataset['break'], break_point=break_point) 
    print(metrics)

    # Para comparar, prueba con una serie estacionaria
    print("\n--- Testing 'stationary' series ---")
    metrics_stat = test_perron(dataset['stationary'], break_point=break_point)
    print(metrics_stat)

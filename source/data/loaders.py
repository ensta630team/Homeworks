import pandas as pd 
import numpy as np

def load_data(path):
    """
    Carga datos desde diferentes formatos de archivo (CSV, Excel, TXT).

    Argumentos:
        path (str): La ruta al archivo de datos.

    Retorna:
        Un DataFrame de pandas, un array de numpy o None si el formato no es compatible.
    """
    # Verifica la extensión del archivo para usar la función de carga correcta.
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        # Se saltan las primeras 2 filas
        df = pd.read_excel(path, skiprows=2)
    elif path.endswith('.txt'):
        df = np.loadtxt(path)
    else:
        # Si el formato no es reconocido, retorna None.
        return None
    
    print('[INFO] ¡Datos cargados exitosamente!')
    return df

def create_dataset(path, problem=6):
    """
    Carga y preprocesa los datos según el problema especificado.

    Argumentos:
        path (str): La ruta al archivo de datos.
        problem (int): El número del problema para determinar el preprocesamiento.
        5 = problema 1 de la tarea 2
        6 = problema 1 de la tarea 3
    Retorna:
        Un diccionario con las series de tiempo (si problem=3) o el DataFrame original (si problem=4).
    """
    # Primero, carga los datos usando la función auxiliar.
    df = load_data(path)

    # Procesamiento específico para el Problema 3 (series de tiempo macroeconómicas).
    if problem == 3:
        # Calcula la tasa de inflación (pi_t) como la diferencia logarítmica del IPC.
        pi_t = np.log(df['IPC'] / df['IPC'].shift(1))
        pi_t.dropna(inplace=True) # Elimina los valores NaN resultantes del shift.
        pi_t = pi_t.to_numpy()
        
        # Calcula el crecimiento del IMACEC (y_t) como la diferencia logarítmica.
        y_t = np.log(df['IMACEC'] / df['IMACEC'].shift(1))
        y_t.dropna(inplace=True)
        y_t = y_t.to_numpy()

        # Obtiene la tasa de política monetaria (i_t) y la convierte a decimal.
        i_t = df['Tasa de política'] / 100
        i_t = i_t.to_numpy()
        
        # Extrae las fechas de la columna 'Periodo'.
        t = df['Periodo'].dt.date
        t = t.to_numpy()

        # Retorna un diccionario con las series de tiempo procesadas.
        return {
            't': t,
            'pi_t': pi_t,
            'y_t': y_t,
            'i_t': i_t
        }
    
    # Para el Problema 4, retorna los datos sin procesar.
    if problem == 4: 
        return df
    
    if problem == 5:
        # Extrae las fechas de la columna 'Periodo'.
        t = df['Periodo'].dt.date
        t = t.to_numpy()

        # Calcula la tasa de inflación (pi_t) como la diferencia logarítmica del IPC.
        pi_t = np.log(df['IPC'] / df['IPC'].shift(12))
        pi_t.dropna(inplace=True) # Elimina los valores NaN resultantes del shift.
        pi_t = pi_t.to_numpy()

        # Calcula el crecimiento del IMACEC (y_t) como la diferencia logarítmica.
        y_t = np.log(df['IMACEC'] / df['IMACEC'].shift(12))
        y_t.dropna(inplace=True)
        y_t = y_t.to_numpy()

        # Obtiene la tasa de política monetaria (i_t) y la convierte a decimal.
        i_t = df['Tasa de política']/100.
        i_t = i_t.to_numpy()

        # Retorna un diccionario con las series de tiempo procesadas.
        return {
            't': t,
            'pi': pi_t,
            'y': y_t,
            'i': i_t
        }
    
    if problem == 6:
        p = np.log(df['IPC'].to_numpy())
        y = np.log(df['IMACEC'].to_numpy())
        i = df['Tasa de política'].to_numpy()/100.

        dtp = df['IPC'].diff(periods= 1).fillna(0.)
        dty = df['IMACEC'].diff(periods= 1).fillna(0.)
        
        t = df['Periodo'].dt.date
        t = t.to_numpy()

        return {
            'p': p,
            'y': y,
            'i': i,
            'dtp': dtp,
            'dty': dty,
            't': t
        }
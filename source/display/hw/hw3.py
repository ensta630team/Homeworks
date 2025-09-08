import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Optional


def configure_matplotlib_for_latex():
    """
    Configura Matplotlib para usar LaTeX si está disponible.
    Si LaTeX no está instalado, deshabilita la opción y usa el renderizado por defecto.
    """
    try:
        # Intenta configurar Matplotlib para usar LaTeX.
        # Esto lanzará un RuntimeError si LaTeX no está instalado.
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        print("LaTeX está disponible y la configuración se ha aplicado.")
    except RuntimeError as e:
        # Si la configuración falla, se asume que LaTeX no está disponible.
        # Se puede establecer la opción 'text.usetex' en False o simplemente omitirla.
        print(f"LaTeX no está disponible. Error: {e}")
        print("Se usará la configuración de texto por defecto.")
        # Opcionalmente, puedes establecer el estilo sin LaTeX
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif",
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}"
        })


FONT_SIZES = {'title': 16, 'label': 14, 'legend': 12, 'tick': 12}


def plot_series(data: dict, 
                fig: Optional[plt.Figure] = None, 
                axes: Optional[plt.Axes] = None,
                title: Optional[str] = None) -> tuple[plt.Figure, np.ndarray]:
    """
    Genera un gráfico con subplots para cada serie de tiempo en el diccionario,
    centrando el último plot si el número de series es impar.

    Args:
        data (dict): Un diccionario que contiene las series de tiempo.
                     Debe incluir la clave 't' (para el tiempo).
        fig (Optional[plt.Figure]): Objeto Figure opcional. Si se proporciona,
                                    los subplots se añadirán a él.

    Returns:
        tuple[plt.Figure, np.ndarray]: Los objetos figure y axes de Matplotlib.
    """
    if 't' not in data:
        raise ValueError("El diccionario de datos debe contener la clave 't' para el tiempo.")

    time_data = data['t']
    series_to_plot = {k: v for k, v in data.items() if k != 't'}
    num_series = len(series_to_plot)
    keys = list(series_to_plot.keys())

    if num_series % 2 != 0:
        nrows = (num_series // 2) + 1
    else:
        nrows = num_series // 2

    if fig is None:
        fig = plt.figure(figsize=(12, 6 * nrows))
    
    locator = mdates.YearLocator(5) 
    # Aplana el array de axes para una fácil iteración
    axes = axes.flatten()
    for i in range(num_series):    
        ax = axes[i]
        ax.plot(time_data, series_to_plot[keys[i]], color='blue', linewidth=1.5)
        ax.set_title(keys[i], fontsize=FONT_SIZES['title'], pad=10)
        
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(locator)
        ax.grid(True)
    
    if title is not None:
        fig.suptitle(title, fontsize=20, y=1.02)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, axes

def plot_dummy_comparison(
    data: pd.DataFrame,
    break_point: int,
    window: int = 10,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Comparación de Variables Dummy',
    xlabel: str = 'Periodo',
    ylabel: str = 'Valor Dummy'
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grafica la comparación de dos series dummy y una línea vertical para un punto de quiebre.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos.
        break_point (int): Posición en el eje x para la línea vertical.
        window (int): Número de periodos a mostrar antes y después del punto de quiebre.
        fig (Optional[plt.Figure]): Figura de Matplotlib existente (opcional).
        ax (Optional[plt.Axes]): Ejes de Matplotlib existentes (opcional).
        title (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.

    Returns:
        tuple[plt.Figure, plt.Axes]: La figura y los ejes del gráfico.
    """
    # Si no se proporcionan ejes (ax), se crea una nueva figura y ejes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        # Si se proporcionan los ejes, se obtiene la figura a la que pertenecen
        fig = ax.get_figure()

    # Definir el rango de datos a graficar usando la ventana
    start = break_point - window
    end = break_point + window
    
    # Graficar las dos series
    ax.plot(data['du'][start:end], linestyle='-', color='k', lw=2, label='Modelo A')
    ax.plot(data['dt_star'][start:end], linestyle='--', color='darkred', label='Modelo B')

    # Añadir la línea vertical
    ax.axvline(x=break_point, color='darkgreen', linestyle=':', label='Break Point')

    # Configurar etiquetas, título y leyenda usando los tamaños de fuente definidos
    ax.set_title(title, fontsize=FONT_SIZES['title'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.set_xticks(range(start, end + 1))


    fig.tight_layout()

    return fig, ax


def plot_series_comparison(
    data: pd.DataFrame,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Comparación de Series de Tiempo',
    xlabel: str = 'Tiempo',
    ylabel: str = 'Valor'
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grafica dos series de tiempo de un DataFrame para su comparación.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos.
        fig (Optional[plt.Figure]): Figura de Matplotlib existente (opcional).
        ax (Optional[plt.Axes]): Ejes de Matplotlib existentes (opcional).
        title (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.

    Returns:
        tuple[plt.Figure, plt.Axes]: La figura y los ejes del gráfico.
    """
    # Si no se proporcionan ejes (ax), se crea una nueva figura y ejes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        # Si se proporcionan los ejes, se obtiene la figura a la que pertenecen
        fig = ax.get_figure()

    # Graficar las dos series con estilos y etiquetas personalizadas
    ax.plot(data['break'], label='Break Random Walk', color='darkblue', linewidth=1.5)
    ax.plot(data['break_ss'], label='Break Estacionaria', color='darkred', linestyle='--', linewidth=1.5)

    # Configurar etiquetas, título y leyenda usando los tamaños de fuente definidos
    ax.set_title(title, fontsize=FONT_SIZES['title'])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True)
    
    fig.tight_layout()

    return fig, ax
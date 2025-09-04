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
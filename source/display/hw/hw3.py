import matplotlib.colors as mcolors
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

def plot_hqc_heatmap(
    data: pd.DataFrame,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Mapa de Calor de Criterio Hannan-Quinn (HQC)',
    xlabel: str = 'Breakpoint',
    ylabel: str = 'Número de Rezagos (p)',
    cmap: str = 'viridis_r',
    quantile_clip: Optional[tuple[float, float]] = (0.05, 0.95)
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grafica una matriz de valores HQC como un mapa de calor y anota el 'p' óptimo.

    Args:
        data (pd.DataFrame): DataFrame con los valores HQC.
        quantile_clip (Optional[tuple[float, float]]): Tupla para recortar la escala de color.
        ... (otros argumentos)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.get_figure()
    
    vmin, vmax = None, None
    if quantile_clip and len(quantile_clip) == 2:
        all_values = data.values.flatten()
        vmin = np.nanquantile(all_values, quantile_clip[0])
        vmax = np.nanquantile(all_values, quantile_clip[1])
        print(f"Escala de color recortada al rango: [{vmin:.2f}, {vmax:.2f}]")

    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation='nearest', vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Valor HQC', fontsize=FONT_SIZES['label'])
    cbar.ax.tick_params(labelsize=FONT_SIZES['tick'])

    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)

    if len(data.columns) > 20:
        ax.set_xticks(np.arange(len(data.columns))[::5])
        ax.set_xticklabels(data.columns[::5], rotation=90)
    if len(data.index) > 20:
        ax.set_yticks(np.arange(len(data.index))[::5])
        ax.set_yticklabels(data.index[::5])

    min_val = data.min().min()
    min_pos = np.where(data == min_val)
    y_idx, x_idx = min_pos[0][0], min_pos[1][0]
    p_optimo = data.index[y_idx]
    bp_optimo = data.columns[x_idx]
    
    rect = plt.Rectangle((x_idx - 0.5, y_idx - 0.5), 1, 1, 
                         edgecolor='red', facecolor='none', lw=3)
    ax.add_patch(rect)
    
    # Añade el texto con el valor de p óptimo al lado derecho del cuadro.
    ax.text(x_idx - x_idx/7, y_idx+5, f"p={p_optimo}",
            color='red',
            fontsize=FONT_SIZES['tick'],
            fontweight='bold',
            ha='left',          # Alineación horizontal a la izquierda
            va='center')       # Alineación vertical al centro

    
    print(f"Óptimo encontrado: HQC={min_val:.2f} en p={p_optimo}, breakpoint={bp_optimo}")

    ax.set_title(title, fontsize=FONT_SIZES['title'], pad=20)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    fig.tight_layout()

    return fig, ax

def plot_series_with_breakpoints(
    data_df: pd.DataFrame, 
    breakpoints_df: pd.DataFrame, 
    variables: List[str],
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = "Análisis de Breakpoint de Perron por Variable"
) -> tuple[plt.Figure, np.ndarray]:
    
    n_vars = len(variables)

    if axes is None or fig is None:
        # Ligeramente ajustado para que se vea bien en grillas (ej. 2x3)
        nrows = (n_vars + 1) // 2 if n_vars > 1 else 1
        ncols = 2 if n_vars > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True, constrained_layout=True)
        if n_vars == 1:
            axes = np.array([axes])
    
    if len(axes.flatten()) < n_vars:
        raise ValueError(f"Se requieren {n_vars} ejes, pero solo se proporcionaron {len(axes.flatten())}.")

    style_map = {
        'model_a': {'color': 'red', 'linestyle': '--', 'label': 'Breakpoint Modelo A'},
        'model_b': {'color': 'dodgerblue', 'linestyle': ':', 'label': 'Breakpoint Modelo B'},
        'model_c': {'color': 'green', 'linestyle': '-.', 'label': 'Breakpoint Modelo C'}
    }

    # --- CAMBIO 1: Loop de ploteo simplificado ---
    for ax, var_name in zip(axes.flatten(), variables):
        # La etiqueta 'label' aquí es solo para la serie, la leyenda final la manejaremos después
        ax.plot(data_df.index, data_df[var_name], color='black', alpha=0.8)
        
        var_breakpoints = breakpoints_df[breakpoints_df['variable'] == var_name]
        
        for _, row in var_breakpoints.iterrows():
            model, break_point_idx = row['model'], int(row['break'])
            break_location = data_df.index[break_point_idx]
            
            style = style_map.get(model, {})
            ax.axvline(x=break_location, color=style.get('color'), linestyle=style.get('linestyle'),
                       label=style.get('label', model), linewidth=2)

        ax.set_title(f'Variable: {var_name.upper()}', loc='left', fontsize=FONT_SIZES['label'])
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    
    # --- CAMBIO 2: Eliminar el subplot vacío si existe ---
    # Si el número de variables es impar y mayor que 1, habrá un subplot vacío
    if n_vars > 1 and n_vars % 2 != 0:
        axes.flatten()[-1].set_visible(False)

    # --- CAMBIO 3: Creación de una leyenda única para toda la figura ---
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Eliminar duplicados
    by_label = dict(zip(labels, handles))
    # Colocar la leyenda en la figura, fuera de los subplots
    fig.legend(by_label.values(), by_label.keys(), 
               loc='upper right', 
               fontsize=FONT_SIZES['legend'],
               bbox_to_anchor=(0.87, 0.5)) # Ajusta la posición si es necesario

    if title:
        fig.suptitle(title, fontsize=FONT_SIZES['title'], weight='bold')
    
    # Esta parte se mantiene igual y ahora funcionará correctamente
    if pd.api.types.is_datetime64_any_dtype(data_df.index):
        fig.autofmt_xdate()
        date_format = mdates.DateFormatter('%Y') # Formato a solo año
        # Aplicar el formato a todos los ejes que estén en la última fila
        # Esto es más robusto para grillas de varias columnas
        num_cols = axes.shape[1] if len(axes.shape) > 1 else 1
        for ax in axes.flatten()[n_vars - num_cols:]:
             if ax.get_visible():
                ax.xaxis.set_major_formatter(date_format)

    return fig, axes
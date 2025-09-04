import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from matplotlib.ticker import MaxNLocator 
from typing import Optional, List


# --- (Se asume la misma configuración de estilo que proporcionaste) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
FONT_SIZES = {'title': 16, 'label': 14, 'legend': 12, 'tick': 12}


def plot_real_vs_generated(
    real_data: np.ndarray, 
    pre_fit_sample: np.ndarray, 
    post_fit_sample: np.ndarray,
    variable_names: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    title: str = r'Comparacion: Serie Real vs. Series Generadas'
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grafica la serie real y las muestras generadas, usando un subplot por cada variable.

    Argumentos:
        real_data (np.ndarray): Array con los datos observados.
        pre_fit_sample (np.ndarray): Array con datos muestreados del modelo no ajustado.
        post_fit_sample (np.ndarray): Array con datos muestreados del modelo ajustado.
        variable_names (List[str], opcional): Nombres para cada variable (columna).
        fig (plt.Figure, opcional): Figura de matplotlib existente.
        axes (np.ndarray, opcional): Array de ejes de matplotlib existentes.
        title (str, opcional): Título general para la figura.

    Retorna:
        tuple[plt.Figure, np.ndarray]: La figura y el array de ejes utilizados.
    """
    n_vars = real_data.shape[1]

    # --- 1. Manejo de Nombres de Variables y Datos ---
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    
    # Convertimos a DataFrame para facilitar el manejo
    real_df = pd.DataFrame(real_data, columns=variable_names)
    pre_fit_df = pd.DataFrame(pre_fit_sample, columns=variable_names)
    post_fit_df = pd.DataFrame(post_fit_sample, columns=variable_names)

    # --- 2. Manejo de Figura y Ejes ---
    if axes is None:
        # Se crean n_vars subplots, compartiendo el eje X
        fig, axes = plt.subplots(
            n_vars, 1, 
            figsize=(14, 5 * n_vars), 
            sharex=True
        )
        # Si solo hay una variable, axes no es un array, lo convertimos para consistencia
        if n_vars == 1:
            axes = np.array([axes])
    else:
        fig = axes.flatten()[0].get_figure()

    # Se establece un título general para toda la figura
    fig.suptitle(title, fontsize=FONT_SIZES['title'])

    # --- 3. Iteración y Trazado en cada Subplot ---
    for i, var_name in enumerate(variable_names):
        ax = axes[i] # Seleccionamos el subplot actual
        
        # Trazar la serie real (línea sólida)
        ax.plot(real_df.index, real_df[var_name], 
                label='Real', color='black', linestyle='-')
        
        # Trazar la muestra antes del ajuste (línea punteada)
        ax.plot(pre_fit_df.index, pre_fit_df[var_name], 
                label='Antes de Ajuste', color='blue', linestyle=':')
        
        # Trazar la muestra después del ajuste (línea discontinua)
        ax.plot(post_fit_df.index, post_fit_df[var_name], 
                label='Despues de Ajuste', color='red', linestyle='--')

        # --- 4. Formato de cada Subplot ---
        ax.set_title(f'{var_name}', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Se añade la etiqueta del eje X solo al último subplot para evitar redundancia
    axes[-1].set_xlabel('Tiempo', fontsize=FONT_SIZES['label'])
    
    return fig, axes

def plot_irf_with_confidence_bands(
    irf_point_estimate: np.ndarray,
    irf_lower_band: np.ndarray,
    irf_upper_band: np.ndarray,
    variable_names: Optional[List[str]] = None,
    impulse_names: Optional[List[str]] = None,
    response_names: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    title: str = r'Funciones de Impulso-Respuesta con Intervalos de Confianza del 95\%'
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grafica las Funciones de Impulso-Respuesta (IRF) con sus bandas de confianza.

    Crea una matriz de subplots (n_vars x n_vars) donde la celda (i, j) muestra la
    respuesta de la variable i a un impulso en la variable j.

    Argumentos:
        irf_point_estimate (np.ndarray): Array de IRF (H, n, n) - Estimación puntual.
        irf_lower_band (np.ndarray): Array (H, n, n) - Límite inferior de la banda.
        irf_upper_band (np.ndarray): Array (H, n, n) - Límite superior de la banda.
        variable_names (List[str], opcional): Nombres para las variables (si impulse y response son iguales).
        impulse_names (List[str], opcional): Nombres de las variables que reciben el impulso.
        response_names (List[str], opcional): Nombres de las variables que responden.
        fig (plt.Figure, opcional): Figura de matplotlib existente.
        axes (np.ndarray, opcional): Array de ejes de matplotlib existentes.
        title (str, opcional): Título general para la figura.

    Retorna:
        tuple[plt.Figure, np.ndarray]: La figura y el array de ejes utilizados.
    """
    horizon, n_vars_response, n_vars_impulse = irf_point_estimate.shape

    # --- 1. Manejo de Nombres de Variables ---
    if variable_names is not None:
        impulse_names = response_names = variable_names
    elif impulse_names is None or response_names is None:
        impulse_names = [f'Impulso en Var {i+1}' for i in range(n_vars_impulse)]
        response_names = [f'Respuesta de Var {i+1}' for i in range(n_vars_response)]

    # --- 2. Manejo de Figura y Ejes ---
    if axes is None:
        fig, axes = plt.subplots(
            n_vars_response, n_vars_impulse,
            figsize=(6 * n_vars_impulse, 5 * n_vars_response),
            sharex=True,
            squeeze=False # Asegura que `axes` siempre sea un array 2D
        )
    else:
        fig = axes.flatten()[0].get_figure()

    fig.suptitle(title, fontsize=FONT_SIZES['title'], y=0.95)

    # --- 3. Iteración y Trazado en cada Subplot ---
    for i in range(n_vars_response):  # Filas -> Variables que responden
        for j in range(n_vars_impulse): # Columnas -> Variables con el shock
            ax = axes[i, j]

            # Extraer la serie para el subplot (i, j)
            point_irf = irf_point_estimate[:, i, j]
            lower_irf = irf_lower_band[:, i, j]
            upper_irf = irf_upper_band[:, i, j]

            # Trazar la estimación puntual
            ax.plot(point_irf, color='black', linestyle='-', label='IRF Puntual')

            # Rellenar el área de la banda de confianza
            ax.fill_between(
                range(horizon), lower_irf, upper_irf,
                color='gray', alpha=0.3, label='Banda de Confianza (95%)'
            )

            # Línea horizontal en cero para referencia
            ax.axhline(0, color='red', linestyle='--', linewidth=1)

            # --- 4. Formato de cada Subplot ---
            ax.set_title(
                f'Impulso: ${impulse_names[j]}$ \n Respuesta: ${response_names[i]}$',
                fontsize=FONT_SIZES['label']
            )
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Poner etiquetas de ejes solo en los bordes
            if i == n_vars_response - 1:
                ax.set_xlabel('Horizonte (Períodos)', fontsize=FONT_SIZES['label'])
            if j == 0:
                ax.set_ylabel('Magnitud', fontsize=FONT_SIZES['label'])

    # Ajustar el layout para evitar solapamientos
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Ajustar rect para dejar espacio al suptitle
    
    return fig, axes

def graficar_criterios_var(
    df_criterios: pd.DataFrame,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    title: str = r'Criterios de Selección de Orden (p) para Modelo VAR'
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grafica los criterios de información AIC, BIC y HQIC en un eje determinado,
    con la leyenda fuera del área de trazado y colores personalizados.

    Argumentos:
        df_criterios (pd.DataFrame): DataFrame con 'p' como índice y columnas AIC, BIC, HQIC.
        fig (plt.Figure, opcional): Figura de matplotlib existente.
        ax (plt.Axes, opcional): Eje de matplotlib existente donde se dibujará el gráfico.
        title (str, opcional): Título para el subplot.

    Retorna:
        tuple[plt.Figure, plt.Axes]: La figura y el eje utilizados.
    """
    # --- 1. Manejo de Figura y Ejes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # --- 2. Trazado de los Criterios con nuevos colores ---
    ax.plot(df_criterios.index, df_criterios['AIC'], marker='o', linestyle='-', color='darkblue', label='AIC')
    ax.plot(df_criterios.index, df_criterios['BIC'], marker='s', linestyle='--', color='darkred', label='BIC')
    ax.plot(df_criterios.index, df_criterios['HQIC'], marker='^', linestyle='-.', color='darkgreen', label='HQIC')

    # --- 3. Marcar los Mínimos con colores correspondientes ---
    p_optimo_bic = df_criterios['BIC'].idxmin()
    ax.axvline(
        p_optimo_bic, color='darkred', linestyle=':', alpha=0.8,
        label=f'BIC Mínimo (p={p_optimo_bic})'
    )
    
    # --- 4. Formato del Gráfico ---
    # ax.set_title(title, fontsize=FONT_SIZES['title'])
    ax.set_xlabel('Número de Rezagos (p)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Valor del Criterio', fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(df_criterios.index)
    
    # --- 5. Manejo de la Leyenda (fuera del eje) ---
    handles, labels = ax.get_legend_handles_labels()
    # Se coloca la leyenda en la figura, fuera del eje 'ax'
    # bbox_to_anchor=(1.02, 1) la sitúa a la derecha y arriba del eje.
    fig.legend(
        handles, labels,
        loc='upper left',
        ncols=4,
        bbox_to_anchor=(0.2, 1.15),
        fontsize=FONT_SIZES['legend'],
        title='Métricas'
    )
    
    # Se ajusta el layout para asegurar que la leyenda no se corte
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # rect deja espacio a la derecha

    return fig, ax

def plot_time_series(
    data: np.ndarray,
    dates: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    title: str = r'Visualización de las Series de Tiempo del Dataset'
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grafica múltiples series de tiempo, con formato de fecha inteligente en el eje X.
    """
    n_vars = data.shape[1]
    
    if dates is not None and len(dates) != data.shape[0]:
        raise ValueError("La longitud del array de fechas debe coincidir con el número de filas en los datos.")

    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]

    if axes is None:
        fig, axes = plt.subplots(
            n_vars, 1,
            figsize=(14, 4 * n_vars),
            sharex=True
        )
        if n_vars == 1:
            axes = np.array([axes])

    # fig.suptitle(title, fontsize=FONT_SIZES['title'])

    for i, var_name in enumerate(variable_names):
        ax = axes[i]
        x_axis = dates if dates is not None else range(data.shape[0])
        ax.plot(x_axis, data[:, i], color='darkblue')
        
        ax.set_title(f'Serie de Tiempopara ${var_name}$', fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Valor', fontsize=FONT_SIZES['label'])

    # --- LÓGICA DE FECHAS MEJORADA ---
    if dates is not None:
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            # Rotar las etiquetas para el eje inferior
            ax.set_xlabel('Fecha', fontsize=FONT_SIZES['label'])
            if i == n_vars - 1:
                plt.setp(ax.get_xticklabels(), rotation=0, ha="right")
            else:
                # Ocultar las etiquetas de los ejes superiores para limpieza
                plt.setp(ax.get_xticklabels(), visible=False)
   
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, axes

def plot_with_user_style(girf_results, variable_names, fig=None, axes=None):
    """
    Grafica la matriz de GIRF utilizando la estructura de bucles y
    títulos proporcionada por el usuario.
    """
    # 1. Obtener dimensiones
    horizonte, k, _ = girf_results.shape

    # 2. Crear la cuadrícula de gráficos (código proporcionado)
    if fig is None or axes is None:
        fig, axes = plt.subplots(k, k, figsize=(5, 4), sharex=True)

    fig.suptitle('Funciones de Respuesta\nal Impulso', fontsize=FONT_SIZES['title'])

    # 3. Iterar para llenar cada subgráfico (código proporcionado)
    for j in range(k):  # Columna: Shock en la variable j
        for i in range(k):  # Fila: Respuesta de la variable i

            # --- ADAPTACIÓN CLAVE ---
            # Se extrae la respuesta de la variable 'i' al shock en la variable 'j'
            # del array 3D estándar. Esto es equivalente a tu 'all_irfs[j][:, i]'.
            response = girf_results[:, i, j]

            ax = axes[i, j]
            ax.plot(range(horizonte), response, marker='.', linestyle='-', color='k')
            ax.axhline(0, color='darkred', linewidth=0.8, linestyle='--')

            # Usamos los nombres de las variables en lugar de los genéricos
            ax.set_title(f'Respuesta de {variable_names[i]} \na Shock en {variable_names[j]}', fontsize=FONT_SIZES['title'])

            # Etiquetas solo en los bordes para mayor claridad
            if i == k - 1:
                ax.set_xlabel('Períodos', fontsize=FONT_SIZES['label'])
            if j == 0:
                ax.set_ylabel('Respuesta', fontsize=FONT_SIZES['label'])

    # 4. Ajustar y mostrar el gráfico (código proporcionado)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes
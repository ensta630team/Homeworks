# Tarea 3: Series de tiempo No Estacionarias

Este repositorio contiene el código y los análisis para la Tarea 3, enfocada en test de estimacion de raices unitarias

![https://github.com/ensta630team/HW2/blob/main/presentation/figures/diagrama.png?raw=true](https://github.com/ensta630team/Homeworks/blob/main/presentation/figures/diagrama.png?raw=true)

### 1. Requisitos

Asegúrate de tener **Python** instalado en tu computador. Puedes descargarlo desde [python.org](https://www.python.org/).

### 2. Instalar las librerías necesarias

Todo lo que necesitas está listado en el archivo `requirements.txt`. Para instalar todo de una vez, abre una terminal o línea de comandos, navega hasta la carpeta de este proyecto y ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el análisis

La forma más fácil de ver los resultados es a través de los Jupyter Notebooks. Estos archivos te permiten ver el código, las explicaciones y los gráficos en un solo lugar.

Navega a la carpeta: `presentation/notebooks/hw_2/`.

Abre los archivos que terminan en `.ipynb` (por ejemplo, `problema_1.ipynb`) usando Jupyter Notebook o Jupyter Lab.

Dentro de los notebooks, puedes ejecutar cada celda de código para replicar los análisis paso a paso.

## Estructura del Repositorio

  * `/data`: Contiene los datos brutos utilizados en los modelos.
  * `/source`: Módulos de Python con las funciones principales para estimar los modelos (VAR, OLS), realizar inferencia estadística y calcular las funciones de impulso-respuesta.
  * `/presentation/notebooks/`: Contiene los Jupyter Notebooks con el desarrollo de cada pregunta de la tarea. Este es el mejor lugar para empezar a explorar.
  * `/presentation/figures/`: Gráficos generados a partir del análisis.
  * `informe.pdf`: El informe final con la explicación detallada de los resultados.

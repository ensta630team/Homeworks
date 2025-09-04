# Guía de Programación para Tests de Raíz Unitaria

Este documento contiene las instrucciones para la programación de los tests de raíz unitaria listados en la tarea. El objetivo es que implementen la lógica de cada test, respetando un formato de función estándar, para que puedan ser evaluados en los notebooks de prueba.


---

### 0. Importante: Actualiza tu Repositorio ⚠️

Antes de comenzar, es **crucial** que te asegures de tener la última versión del repositorio para evitar conflictos. Sigue estos pasos para actualizar y crear tu rama de trabajo desde VS Code:

1.  **Abre el panel de control de Git:** Haz clic en el icono de Git en la barra lateral de VS Code (el tercer icono desde arriba).
2.  **Sincroniza los cambios:** En la parte superior del panel de Git, haz clic en el icono de los puntos suspensivos (...) y selecciona 'Pull, Push' -> 'Pull from...' y luego elige 'origin/main' o la rama principal del proyecto. Esto descargará todos los cambios recientes.
3.  **Crea tu nueva rama:** Una vez que el 'pull' haya terminado, haz clic en el nombre de la rama actual en la esquina inferior izquierda de la ventana de VS Code. Se abrirá un menú. Elige '+ Create new branch...'.
4.  **Nombra tu rama:** Escribe un nombre descriptivo para tu rama (por ejemplo, 'feature/kpss-test'). Presiona 'Enter'.
5.  **Comienza a trabajar:** Ahora estás en tu propia rama de trabajo y puedes empezar a programar sin afectar la rama principal.

---

---

### 1. Entorno de Trabajo

Antes de comenzar a programar, asegúrate de tener el entorno configurado. Puedes instalar las librerías necesarias con el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```
---

### 2. Archivos de Trabajo

El código de los tests debe ser escrito en la carpeta `source/fit/inference/`. Dentro de esta carpeta, cada test tiene un archivo `.py` asignado con su nombre.

* **Tu tarea es intervenir el archivo que corresponde al test que te fue asignado.** Por ejemplo, si te toca programar el test de KPSS, debes ir al archivo `source/fit/inference/kpss.py` y escribir tu código allí.

---

### 3. Estándar de la Función

Cada test debe ser programado como una función que siga el siguiente formato estándar:

* **Entrada:** Debe recibir, al menos, un array de NumPy (`np.ndarray`) con la serie de tiempo. Si el test requiere parámetros adicionales (por ejemplo, el número de rezagos o el tipo de regresión), debes incluirlos como argumentos opcionales.
* **Salida:** Debe retornar un `DataFrame` de pandas con las métricas del test. Para mantener la uniformidad, el DataFrame debe tener al menos dos columnas con los siguientes nombres: `Métrica` y `Valor`.

Para el test de Dickey-Fuller, la firma de la función sería:

def test_df(serie: np.ndarray) -> pd.DataFrame:
    # ... tu lógica aquí
    return df_metrics

El archivo `source/fit/inference/ljungbox.py` está programado y puedes usarlo como **molde de referencia** para entender la estructura y el formato de salida.

---

### 4. Nombres de las Métricas

Para una correcta evaluación, las filas del DataFrame de salida deben tener nombres estandarizados en la columna `Métrica`. Se recomiendan los siguientes:

* `Estadístico` (valor del estadístico del test)
* `Valor P` (p-valor del test)
* `Valores Críticos` (si aplica)

Si tu test tiene métricas adicionales, siéntete libre de agregarlas, siempre manteniendo la estructura de `Métrica` y `Valor`.

---

### 5. Validación

Para probar tu código, puedes usar los notebooks que se encuentran en `presentation/notebooks/hw_3/`.

* `testing.ipynb`: Este notebook contiene simulaciones de series de tiempo sintéticas. Puedes usarlo para verificar si tu test detecta correctamente las propiedades de cada serie.
* `pregunta.ipynb`: Este notebook evalúa los tests con series de tiempo reales, lo cual te permite ver cómo se comportan tus métodos en la práctica.

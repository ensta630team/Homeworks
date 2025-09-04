import numpy as np
import random
import os

def go_to_project_root():
    """
    Cambia el directorio de trabajo actual (CWD) al directorio raíz del proyecto.
    
    Busca un archivo 'marcador' (ej. 'README.md') subiendo en la jerarquía de
    directorios. Si lo encuentra, establece ese directorio como el CWD.
    Esto es útil para que las rutas relativas funcionen correctamente.
    """
    
    # Define el archivo que identifica la raíz del proyecto.
    # Puedes cambiarlo por otro archivo como '.git' o 'pyproject.toml'.
    marker_file = "README.md"
    
    # Obtiene el directorio de trabajo actual desde donde se ejecuta el script.
    current_directory = os.getcwd()
    
    # Inicia un bucle para subir por el árbol de directorios.
    while True:
        # Comprueba si el archivo marcador existe en el directorio actual.
        if marker_file in os.listdir(current_directory):
            # Si se encuentra, cambia el CWD a este directorio y termina la función.
            os.chdir(current_directory)
            print(f"CWD cambiado a la raíz del proyecto: {os.getcwd()}")
            return
        
        # Si no se encuentra, sube al directorio padre.
        parent_directory = os.path.dirname(current_directory)
        
        # Condición de parada: se ha llegado a la raíz del sistema de archivos.
        if parent_directory == current_directory:
            print("No se encontró el marcador de la raíz del proyecto. El CWD no se cambió.")
            return
        
        # Actualiza el directorio para la siguiente iteración del bucle.
        current_directory = parent_directory



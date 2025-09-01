import numpy as np 

from statsmodels.tsa.stattools import adfuller


def test_stationarity(df_simulated):
    for col in df_simulated.columns:
        result = adfuller(df_simulated[col])
        print(f"Variable: {col}")
        print(f"Estadístico ADF: {result[0]:.4f}")
        print(f"  Valor p: {result[1]:.4f}")
        print(f"  Valores críticos: {result[4]}")
        if result[1] <= 0.05:
            print("  La serie es estacionaria (se rechaza la hipótesis nula de raíz unitaria)")
        else:
            print("  La serie NO es estacionaria (no se puede rechazar la hipótesis nula de raíz unitaria)")
        print("-" * 30)


def get_GIRD(sigma_u, n_periods, k, psi_coeffs):
    girf = np.zeros((n_periods + 1, k, k))
    for j in range(k):
        e_j = np.zeros(k)
        e_j[j] = 1.0

        if sigma_u[j, j] == 0: # Ahora sigma_u es un array de NumPy, el acceso es correcto.
            print(f"Advertencia: La varianza del error para la variable {j+1} es cero. No se puede calcular GIRF para este shock.")
            continue

        normalizing_factor = 1.0 / np.sqrt(sigma_u[j, j])
        for h in range(n_periods + 1):
            if h == 0:
                girf[h, :, j] = np.zeros(k)
            else:
                term1 = normalizing_factor * psi_coeffs[h-1]
                term2 = np.dot(term1, sigma_u)
                girf[h, :, j] = np.dot(term2, e_j)
    return girf

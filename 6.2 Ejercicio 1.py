import numpy as np
import matplotlib.pyplot as plt

# Definición de la EDO: dT/dx = -0.25(T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, T0, x_end, h):
    x_vals = [x0]
    T_vals = [T0]
    
    x = x0
    T = T0
    
    print(f"{'x':>10} {'T':>15}")
    print(f"{x:10.4f} {T:15.6f}")
    
    while x < x_end:
        k1 = f(x, T)
        k2 = f(x + h/2, T + h/2 * k1)
        k3 = f(x + h/2, T + h/2 * k2)
        k4 = f(x + h, T + h * k3)
        
        T += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        x_vals.append(x)
        T_vals.append(T)
        
        print(f"{x:10.4f} {T:15.6f}")
    
    return x_vals, T_vals

# Parámetros iniciales
x0 = 0         # Distancia inicial (m)
T0 = 100       # Temperatura inicial (°C)
x_end = 2      # Distancia final (m)
h = 0.1        # Tamaño del paso

# Llamada al método de Runge-Kutta
x_vals, T_vals = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta: T(x) = 25 + 75 * e^(-0.25 * x)
x_exact = np.linspace(x0, x_end, 200)
T_exact = 25 + 75 * np.exp(-0.25 * x_exact)

# Graficar la solución numérica y la exacta
plt.figure(figsize=(8,5))
plt.plot(x_vals, T_vals, 'bo-', label="Solución RK4 (Numérica)")
plt.plot(x_exact, T_exact, 'g-', label="Solución Exacta", linewidth=2)
plt.xlabel("Distancia (x) [m]")
plt.ylabel("Temperatura (T) [°C]")
plt.title("Transferencia de Calor en un Tubo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("transferencia_calor.png")
plt.show()
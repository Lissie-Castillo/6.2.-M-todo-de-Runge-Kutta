import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones diferenciales
def f(t, y1, y2):
    dy1_dt = y2
    dy2_dt = -2*y2 - 5*y1
    return dy1_dt, dy2_dt

# Método de Runge-Kutta de 4to orden para sistemas de primer orden
def runge_kutta_4_system(f, t0, y1_0, y2_0, t_end, h):
    t_vals = [t0]
    y1_vals = [y1_0]
    y2_vals = [y2_0]
    
    t = t0
    y1 = y1_0
    y2 = y2_0
    
    while t < t_end:
        k1y1, k1y2 = f(t, y1, y2)
        k2y1, k2y2 = f(t + h/2, y1 + h/2 * k1y1, y2 + h/2 * k1y2)
        k3y1, k3y2 = f(t + h/2, y1 + h/2 * k2y1, y2 + h/2 * k2y2)
        k4y1, k4y2 = f(t + h, y1 + h * k3y1, y2 + h * k3y2)
        
        y1 += h * (k1y1 + 2*k2y1 + 2*k3y1 + k4y1) / 6
        y2 += h * (k1y2 + 2*k2y2 + 2*k3y2 + k4y2) / 6
        t += h
        
        t_vals.append(t)
        y1_vals.append(y1)
        y2_vals.append(y2)
    
    return t_vals, y1_vals, y2_vals

# Parámetros iniciales
t0 = 0
y1_0 = 1  # posición inicial
y2_0 = 0  # velocidad inicial
t_end = 5
h = 0.1

# Resolver el sistema usando Runge-Kutta
t_vals, y1_vals, y2_vals = runge_kutta_4_system(f, t0, y1_0, y2_0, t_end, h)

# Mostrar los resultados en una tabla
print(f"{'t (s)':>10} {'y1 (Posición)':>15} {'y2 (Velocidad)':>15}")
for i in range(len(t_vals)):
    print(f"{t_vals[i]:10.2f} {y1_vals[i]:15.6f} {y2_vals[i]:15.6f}")

# Graficar la trayectoria de la masa
plt.figure(figsize=(8,5))
plt.plot(t_vals, y1_vals, 'b-', label='Posición (y1)')
plt.plot(t_vals, y2_vals, 'r-', label='Velocidad (y2)')
plt.xlabel("Tiempo (s)")
plt.ylabel("Magnitud")
plt.title("Dinámica de un resorte amortiguado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
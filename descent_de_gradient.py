
import matplotlib.pyplot as plt
import numpy as np

#f(x):
def f(x):
    return (x - 1)**2 * (x - 2)**2
def df(x):
    return 2 * (x - 1) * (x - 2)**2 + 2 * (x - 1)**2
# Méthode de DG
def gradient_descent_fixed_step(f, df, x0, alpha, epsilon, max_iter):
    x = x0
    iterations = 0
    while iterations < max_iter and abs(df(x)) > epsilon:
        x = x - alpha * df(x)
        iterations += 1
    return x, f(x), iterations

x0 = 0
alpha = 0.01
epsilon = 1e-5
max_iter = 1000

# DG pour f(x)=(x - 1)^2 * (x - 2)^2
xmin, fmin, num_iterations = gradient_descent_fixed_step(
    f, df, x0, alpha, epsilon, max_iter)

print(
    f"Minimum trouvé : x = {xmin}, f(xmin) = {fmin}, nombre d'itérations = {num_iterations}")

#Partie 2

def f(x):
    return (x - 2)**2

def df_f(x):
    return 2 * (x - 2)

def g(x):
    return -np.exp(-(x - 1)**2) * (x - 2)**2

def df_g(x):
    return -2 * (x - 2) * np.exp(-(x - 1)**2) + (x - 2)**2 * np.exp(-(x - 1)**2)

def gradient_descent(f, df, x0, alpha, epsilon, max_iter):
    x = x0
    iterations = 0
    x_history = [x]
    f_history = [f(x)]

    while iterations < max_iter and abs(df(x)) > epsilon:
        x = x - alpha * df(x)
        iterations += 1
        x_history.append(x)
        f_history.append(f(x))

    return x, f(x), iterations, x_history, f_history

x0 = 0
alpha = 0.01
epsilon = 1e-5
max_iter = 1000

# Appliquer DG à f(x)
xmin_f, fmin_f, num_iterations_f, x_history_f, f_history_f = gradient_descent(
    f, df_f, x0, alpha, epsilon, max_iter)

# Appliquer DG à g(x)
xmin_g, fmin_g, num_iterations_g, x_history_g, f_history_g = gradient_descent(
    g, df_g, x0, alpha, epsilon, max_iter)

# Afficher results pour f(x)
print(
    f"Minimum trouvé pour f(x) : x = {xmin_f}, f(xmin) = {fmin_f}, nombre d'itérations = {num_iterations_f}")

# Afficher results pour g(x)
print(
    f"Minimum trouvé pour g(x) : x = {xmin_g}, f(xmin) = {fmin_g}, nombre d'itérations = {num_iterations_g}")

# Tracer les fonctions et les chemins pris par l'algorithme
x_values = np.linspace(-1, 5, 400)
y_f = f(x_values)
y_g = g(x_values)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_values, y_f, label="f(x)")
plt.scatter(xmin_f, fmin_f, color='red', marker='o', label="Minimum trouvé")
plt.plot(x_history_f, f_history_f,
         label="Chemin de la descente de gradient", linestyle='--')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Fonction f(x)")
plt.subplot(1, 2, 2)
plt.plot(x_values, y_g, label="g(x)")
plt.scatter(xmin_g, fmin_g, color='red', marker='o', label="Minimum trouvé")
plt.plot(x_history_g, f_history_g,
         label="Chemin de la descente de gradient", linestyle='--')
plt.xlabel("x")
plt.ylabel("g(x)")
plt.legend()
plt.title("Fonction g(x)")
plt.tight_layout()
plt.show()
x0_values = [0, 5]
alpha_values = [0.1, 0.5]

fig, axes = plt.subplots(len(x0_values), len(alpha_values), figsize=(12, 8))
fig.suptitle("Descente de gradient pour différentes valeurs de x0 et alpha")

for i, x0 in enumerate(x0_values):
    for j, alpha in enumerate(alpha_values):
        # Appliquer DG à f(x)
        xmin_f, fmin_f, num_iterations_f, x_history_f, f_history_f = gradient_descent(
            f, df_f, x0, alpha, epsilon, max_iter)

        # Appliquer DG à g(x)
        xmin_g, fmin_g, num_iterations_g, x_history_g, f_history_g = gradient_descent(
            g, df_g, x0, alpha, epsilon, max_iter)
        
        x_values = np.linspace(-1, 5, 400)
        y_f = f(x_values)
        y_g = g(x_values)

        ax = axes[i, j]
        ax.plot(x_values, y_f, label="f(x)")
        ax.scatter(xmin_f, fmin_f, color='red',
                   marker='o', label="Minimum trouvé (f)")
        ax.plot(x_history_f, f_history_f,
                label="Chemin de la descente (f)", linestyle='--')
        ax.plot(x_values, y_g, label="g(x)")
        ax.scatter(xmin_g, fmin_g, color='green',
                   marker='s', label="Minimum trouvé (g)")
        ax.plot(x_history_g, f_history_g,
                label="Chemin de la descente (g)", linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"x0 = {x0}, alpha = {alpha}")
        ax.legend()

plt.tight_layout()
plt.show()

# Le choix de la valeur initiale de x0 et de α affecte la vitesse et la stabilité de convergence de l'algorithme de descente de gradient
# x0 proche du minimum global et α approprié permettent une convergence rapide, mais des choix inappropriés peuvent entraîner une convergence lente ou une divergence.


# Méthode DG avec inertie
def gradient_descent_with_momentum(f, df, x0, alpha_min, alpha_max, epsilon, max_iter, beta):
    x = x0
    alpha = alpha_min
    iterations = 0
    x_history = [x]
    f_history = [f(x)]
    gradient = df(x)
    v = 0  

    while iterations < max_iter and np.linalg.norm(gradient) > epsilon:
        x_new = x - alpha * gradient + beta * v
        gradient_new = df(x_new)
        if np.dot(gradient, gradient_new) > 0:
            alpha = min(1.5 * alpha, alpha_max)
        else:
            alpha = alpha / 2
        v = beta * v + alpha * gradient_new
        x = x_new
        gradient = gradient_new
        iterations += 1
        x_history.append(x)
        f_history.append(f(x))
    return x, f(x), iterations, x_history, f_history


alpha_min = 0.0001
alpha_max = 1.0
beta = 0.5

# Appliquer DG avec inertie à f(x)
xmin_f_momentum, fmin_f_momentum, num_iterations_f_momentum, x_history_f_momentum, f_history_f_momentum = gradient_descent_with_momentum(
    f, df_f, x0, alpha_min, alpha_max, epsilon, max_iter, beta)
# Appliquer DG avec inertie à g(x)
xmin_g_momentum, fmin_g_momentum, num_iterations_g_momentum, x_history_g_momentum, f_history_g_momentum = gradient_descent_with_momentum(
    g, df_g, x0, alpha_min, alpha_max, epsilon, max_iter, beta)
# Tracer les fonctions et les chemins pris par l'algorithme avec inertie
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_values, y_f, label="f(x)")
plt.scatter(xmin_f_momentum, fmin_f_momentum, color='red',
            marker='o', label="Minimum trouvé (f)")
plt.plot(x_history_f_momentum, f_history_f_momentum,
         label="Chemin de la descente (f avec inertie)", linestyle='--')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Fonction f(x) avec inertie")
plt.subplot(1, 2, 2)
plt.plot(x_values, y_g, label="g(x)")
plt.scatter(xmin_g_momentum, fmin_g_momentum, color='green',
            marker='s', label="Minimum trouvé (g)")
plt.plot(x_history_g_momentum, f_history_g_momentum,
         label="Chemin de la descente (g avec inertie)", linestyle='--')
plt.xlabel("x")
plt.ylabel("g(x)")
plt.legend()
plt.title("Fonction g(x) avec inertie")
plt.tight_layout()
plt.show()

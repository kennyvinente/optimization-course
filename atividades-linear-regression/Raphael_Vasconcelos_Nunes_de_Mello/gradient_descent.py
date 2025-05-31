import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

def gradient_descent(f, grad_f, x0, alpha_type='fixed', alpha=0.01,
                     max_steps=1000, tolerance=0.0001,
                     c1=1e-4, rho=0.5, max_line_search_iter=20):
    X = np.array(x0, dtype=float)
    f_values = [f(X)]
    path = [X.copy()]
    num_steps = 0

    for step in range(max_steps):
        grad = grad_f(X)
        if np.linalg.norm(grad) < tolerance:
            break

        if alpha_type == 'fixed':
            step_size = alpha
        elif alpha_type == 'backtracking':
            step_size = alpha
            for _ in range(max_line_search_iter):
                new_X_try = X - step_size * grad
                if f(new_X_try) <= f(X) - c1 * step_size * np.dot(grad, grad):
                    break
                step_size *= rho
            if f(X - step_size * grad) > f(X) - c1 * step_size * np.dot(grad, grad):
                 pass
        else:
            raise ValueError("alpha_type must be 'fixed' or 'backtracking'")

        X_new = X - step_size * grad
        
        if np.linalg.norm(X_new - X) < tolerance:
            X = X_new
            path.append(X.copy())
            f_values.append(f(X))
            num_steps += 1
            break
            
        X = X_new
        path.append(X.copy())
        f_values.append(f(X))
        num_steps += 1

    return X, f_values, path, num_steps

def f(X):
    x, y = X
    return (x-47)**2 + (y-0.1)**2 + 2

def gradf(X):
    x, y = X
    return np.array([2*(x-47), 2*(y-0.1)])

def plot_convergence_path(f_func, path_data, title, x_range=[-10, 60], y_range=[-5, 5]):
    path_points_for_plot = np.array(path_data)
    df_path_for_plot = pd.DataFrame(path_points_for_plot, columns=['x', 'y'])

    x_coords = np.linspace(x_range[0], x_range[1], 400)
    y_coords = np.linspace(y_range[0], y_range[1], 400)
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
    
    Z_mesh = f_func([X_mesh, Y_mesh])


    plt.figure(figsize=(8, 6))
    cp = plt.contour(X_mesh, Y_mesh, Z_mesh, levels=30, cmap='viridis')
    plt.colorbar(cp)
    
    if not df_path_for_plot.empty:
        plt.plot(df_path_for_plot['x'], df_path_for_plot['y'], marker='o', color='red', linestyle='-', label='Path')
        plt.scatter(df_path_for_plot['x'].iloc[0], df_path_for_plot['y'].iloc[0], color='blue', s=100, zorder=5, label='Start')
        plt.scatter(df_path_for_plot['x'].iloc[-1], df_path_for_plot['y'].iloc[-1], color='green', s=100, zorder=5, label='End')
        
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

x0 = np.array([0, 0])
result = gradient_descent(f, gradf, x0)

path_points_initial = np.array(result[2])
df_path_initial = pd.DataFrame(path_points_initial, columns=['x', 'y'])
print("DataFrame from the first gradient descent path:")
print(df_path_initial.head())

plot_convergence_path(f, result[2], "Gradient Descent Path for f", x_range=[-10, 60], y_range=[-5, 5])

def f_example(X):
    x, y = X
    return (x-47)**2 + (y-0.1)**2 + 2

def grad_f_example(X):
    x, y = X
    return np.array([2*(x-47), 2*(y-0.1)])

result_example = gradient_descent(f_example, grad_f_example, [0, 0], alpha_type='fixed', alpha=0.01)
plot_convergence_path(f_example, result_example[2], "Gradient Descent for f_example", x_range=[-10, 60], y_range=[-5, 5])

def func(X_rosen):
    x_rosen, y_rosen = X_rosen
    return 20*(y_rosen - x_rosen**2)**2 + (1 - x_rosen)**2

def grad(X_rosen):
    value = np.zeros(len(X_rosen))
    xd = np.array(X_rosen, dtype=float)
    h = 10**-6
    
    fx = func(X_rosen)

    for j in range(len(X_rosen)):
        original_val_j = X_rosen[j]
        xd[j] = original_val_j + h
        value[j] = (func(xd) - fx)/h
        xd[j] = original_val_j
    return value

result_rosen_fixed = gradient_descent(func, grad, [0,0], alpha_type='fixed', alpha=0.001, max_steps=10000, tolerance=1e-5)
plot_convergence_path(func, result_rosen_fixed[2], "Gradient Descent for Rosenbrock (Fixed Alpha=0.001)", x_range = [-2,2], y_range = [-1,3])

result_rosen_backtracking = gradient_descent(func, grad, [0,0], alpha_type='backtracking', alpha=1.0, max_steps=10000, tolerance=1e-5)
plot_convergence_path(func, result_rosen_backtracking[2], "Gradient Descent for Rosenbrock (Backtracking)", x_range = [-2,2], y_range = [-1,3])
#------------------------
# Informações do Autor
#------------------------
# Nome: Jonathas Tavares Neves
# Matrícula: 3250122
# Curso: Doutorado em Engenharia Elétrica - PPGEE
# Disciplina: Otimização

#------------------------
# Importando pacotes
#------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

# Mudando o estilo dos gráficos para melhor visualização
plt.style.use('seaborn-v0_8-whitegrid')

#------------------------
# Definição das funções
#------------------------

# Função objetivo: representa um paraboloide deslocado. Nosso objetivo é minimizar essa função.
def f(X):
    x = np.asarray(X[0])
    y = np.asarray(X[1])
    return (x - 47)**2 + (y - 0.1)**2 + 2

# Gradiente da função objetivo: vetor de derivadas parciais em relação a x e y.
def gradf(X):
    return np.array([2*(X[0] - 47), 2*(X[1] - 0.1)])

#------------------------
# Visualização 3D da função
#------------------------
# Gera gráficos 3D e de contorno da função f(x,y) para observar sua forma

def plot_function_3d():
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    x_range = np.linspace(45, 49, 50)
    y_range = np.linspace(-2, 2, 50)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    Z_mesh = f([X_mesh, Y_mesh])
    ax1.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Gráfico 3D de f(x,y)')

    ax2 = fig.add_subplot(122)
    levels = np.logspace(0.3, 3, 20)
    contour = ax2.contour(X_mesh, Y_mesh, Z_mesh, levels=levels)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contorno de f(x,y)')
    plt.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.show()

#------------------------
# Gradiente Descendente
#------------------------
# Algoritmo principal de otimização, que realiza iterações para encontrar o mínimo da função.

def gradient_descent(f, grad_f, x0, alpha_type='fixed', alpha=0.01,
                     max_steps=1000, tolerance=1e-4,
                     c1=1e-4, rho=0.5, max_line_search_iter=20):
    x_current = np.array(x0, dtype=float)
    path = [x_current.copy()]  # guarda o caminho percorrido
    f_values = [f(x_current)]  # valores da função a cada passo
    gradient_norms = []  # norma do gradiente a cada passo (indicador de convergência)

    for step in range(max_steps):
        gradient = grad_f(x_current)
        grad_norm = np.linalg.norm(gradient)
        gradient_norms.append(grad_norm)

        # Critério de convergência
        if grad_norm < tolerance:
            print(f"Convergiu após {step} iterações (||grad|| = {grad_norm:.2e})")
            break

        # Cálculo do passo
        if alpha_type == 'fixed':
            step_size = alpha
        elif alpha_type == 'backtracking':
            step_size = backtracking_line_search(f, grad_f, x_current, gradient,
                                                 alpha, c1, rho, max_line_search_iter)
        else:
            raise ValueError("alpha_type deve ser 'fixed' ou 'backtracking'")

        # Atualiza a posição
        x_current = x_current - step_size * gradient
        path.append(x_current.copy())
        f_values.append(f(x_current))

    return x_current, f_values, path, step + 1, gradient_norms

#------------------------
# Busca de linha com Backtracking
#------------------------
# Estratégia adaptativa para determinar o tamanho do passo (alpha)

def backtracking_line_search(f, grad_f, x, gradient, alpha_init, c1, rho, max_iter):
    alpha = alpha_init
    f_x = f(x)
    grad_norm_sq = np.dot(gradient, gradient)

    for _ in range(max_iter):
        x_new = x - alpha * gradient
        f_new = f(x_new)

        if f_new <= f_x - c1 * alpha * grad_norm_sq:
            return alpha
        alpha *= rho

    return alpha

#------------------------
# Visualizações dos resultados
#------------------------
# Mostra o caminho percorrido na função e os gráficos da função e da norma do gradiente

def plot_convergence_path(func, path, title, x_range=None, y_range=None, levels=20):
    path = np.array(path)
    if x_range is None:
        x_range = [path[:, 0].min() - 1, path[:, 0].max() + 1]
    if y_range is None:
        y_range = [path[:, 1].min() - 1, path[:, 1].max() + 1]

    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=levels, alpha=0.7)
    plt.colorbar(contour)
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=8, label='Caminho')
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Início')
    plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='Fim')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence_analysis(f_values, gradient_norms, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(f_values, 'b-', linewidth=2)
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Valor da função')
    ax1.set_title(f'{title} - Valores da Função')
    ax1.set_yscale('log')
    ax1.grid(True)

    ax2.plot(gradient_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Norma do Gradiente')
    ax2.set_title(f'{title} - Norma do Gradiente')
    ax2.set_yscale('log')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

#------------------------
# Exportar Resultados para CSV
#------------------------
# Salva os resultados da otimização em um arquivo CSV para análise posterior

def export_optimization_results(path, f_values, gradient_norms, filename='resultados_otimizacao.csv'):
    df = pd.DataFrame(path, columns=['x', 'y'])
    df['f(x,y)'] = f_values
    grad_series = pd.concat([pd.Series(gradient_norms), pd.Series([np.nan])], ignore_index=True)
    df['||grad||'] = grad_series
    df.to_csv(filename, index=False)
    print(f"Resultados exportados para {filename}")

#------------------------
# Engenharia de Features (Exemplo)
#------------------------
# Normaliza as variáveis para evitar escalas diferentes

def feature_engineering(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

#------------------------
# Execução Comparativa com Visualizações
#------------------------
# Roda os testes com passo fixo e com busca de linha adaptativa (backtracking)

def compare_methods():
    x0 = [50.0, 2.0]  # ponto inicial

    print("Executando com passo fixo...")
    start = time.time()
    _, fvals_fixed, path_fixed, _, gradnorms_fixed = gradient_descent(
        f, gradf, x0, alpha_type='fixed', alpha=0.1)
    print(f"Tempo: {time.time() - start:.4f}s\n")

    print("Executando com backtracking...")
    start = time.time()
    _, fvals_bt, path_bt, _, gradnorms_bt = gradient_descent(
        f, gradf, x0, alpha_type='backtracking', alpha=1.0)
    print(f"Tempo: {time.time() - start:.4f}s\n")

    plot_convergence_path(f, path_fixed, 'Gradiente Descendente - Passo Fixo')
    plot_convergence_analysis(fvals_fixed, gradnorms_fixed, 'Passo Fixo')
    export_optimization_results(path_fixed, fvals_fixed, gradnorms_fixed, 'passo_fixo.csv')

    plot_convergence_path(f, path_bt, 'Gradiente Descendente - Backtracking')
    plot_convergence_analysis(fvals_bt, gradnorms_bt, 'Backtracking')
    export_optimization_results(path_bt, fvals_bt, gradnorms_bt, 'backtracking.csv')

#------------------------
# Execução Principal
#------------------------

if __name__ == '__main__':
    plot_function_3d()
    compare_methods()

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
import os

#------------------------
# Pré-processamento e utilidades
#------------------------
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def computeCost(X, y, theta):
    m = len(y)
    J = (1/(2*m)) * np.sum((X @ theta - y)**2)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    grad_norm_history = []

    for i in range(num_iters):
        grad = (1/m) * X.T @ (X @ theta - y)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        theta = theta - alpha * grad
        J_history.append(computeCost(X, y, theta))

        # Condições de Wolfe (simplificadas)
        if grad_norm < 1e-4:
            print(f"Convergência atingida na iteração {i}")
            break

    return theta, J_history, grad_norm_history

#------------------------
# Caminho dos dados (corrigido)
#------------------------
data_dir = os.path.dirname(__file__)
file_ex1data1 = os.path.join(data_dir, 'ex1data1.txt')

#------------------------
# Carregando e normalizando os dados
#------------------------
data = np.loadtxt(file_ex1data1, delimiter=',')
X = data[:, 0].reshape(-1, 1)  # Tamanho da população
y = data[:, 1].reshape(-1, 1)  # Lucro
m = len(y)

# Engenharia de Features: adicionando termo de bias (1)
X_norm, mu, sigma = featureNormalize(X)
X_prep = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# Inicialização dos parâmetros
theta = np.zeros((2, 1))
alpha = 0.01
num_iters = 400

# Descida do gradiente
theta, J_history, grad_norm_history = gradientDescent(X_prep, y, theta, alpha, num_iters)

#------------------------
# Gráficos e Resultados
#------------------------
plt.figure(figsize=(8, 6))
plt.plot(J_history)
plt.xlabel('Iterações')
plt.ylabel('Custo J')
plt.title('Evolução da Função Custo')
plt.grid(True)
plt.savefig("custo_vs_iteracoes.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(grad_norm_history)
plt.xlabel('Iterações')
plt.ylabel('Norma do Gradiente')
plt.title('Norma do Gradiente ao Longo das Iterações')
plt.grid(True)
plt.savefig("gradiente_vs_iteracoes.png")
plt.show()

# Gráfico da regressão linear com dados normalizados
x_vals = np.linspace(min(X_norm), max(X_norm), 100)
x_vals_prep = np.c_[np.ones(100), x_vals]
y_vals_pred = x_vals_prep @ theta

plt.figure(figsize=(8, 6))
plt.scatter(X_norm, y, label='Dados', c='blue', marker='x')
plt.plot(x_vals, y_vals_pred, label='Regressão Linear', c='green')
plt.xlabel('População Normalizada')
plt.ylabel('Lucro')
plt.title(f'Ajuste por Regressão Linear (θ₀={theta[0,0]:.2f}, θ₁={theta[1,0]:.2f})')
plt.legend()
plt.grid(True)
plt.savefig("regressao_normalizada.png")
plt.show()

# Gráfico da regressão linear nos dados originais (não normalizados)
x_vals_orig = np.linspace(min(X), max(X), 100)
x_vals_norm = (x_vals_orig - mu) / sigma
x_vals_prep_orig = np.c_[np.ones(100), x_vals_norm]
y_vals_pred_orig = x_vals_prep_orig @ theta

plt.figure(figsize=(8, 6))
plt.scatter(X, y, c='blue', marker='x', label='Dados Originais')
plt.plot(x_vals_orig, y_vals_pred_orig, c='green', label='Regressão Linear')
plt.xlabel('População (milhares)')
plt.ylabel('Lucro ($10.000)')
plt.title('Regressão Linear (Dados Originais)')
plt.legend()
plt.grid(True)
plt.savefig("regressao_original.png")
plt.show()

# Detecção de outliers via SVM (One-Class)
from sklearn.svm import OneClassSVM

def detectar_outliers(X, y):
    dados = np.hstack((X, y))  # Certifique-se que X e y estão concatenados corretamente
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
    pred = svm.fit_predict(dados)
    return pred == -1

outliers = detectar_outliers(X_norm, y)

plt.figure(figsize=(8, 6))
plt.scatter(X_norm[~outliers], y[~outliers], marker='x', c='blue', label='Dados válidos')
plt.scatter(X_norm[outliers], y[outliers], marker='o', facecolors='none', edgecolors='r', s=100, label='Outliers')
plt.plot(x_vals, y_vals_pred, label='Regressão Linear', c='green')
plt.xlabel('População Normalizada')
plt.ylabel('Lucro')
plt.title('Regressão Linear com Outliers Destacados (SVM)')
plt.legend()
plt.grid(True)
plt.savefig("outliers_svm.png")
plt.show()

#------------------------
# Visualização da função custo em função de theta
#------------------------
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Calculando o custo J(θ) para cada par de (θ₀, θ₁)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost(X_prep, y, t)

# Transpondo J_vals para plot correto com matplotlib
J_vals = J_vals.T

# Gráfico de superfície 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(T0, T1, J_vals, cmap='viridis')
ax.set_xlabel('θ₀')
ax.set_ylabel('θ₁')
ax.set_zlabel('Custo J(θ)')
ax.set_title('Superfície da Função Custo J(θ)')
plt.savefig("superficie_custo_theta.png")
plt.show()

# Gráfico de contorno (contour)
plt.figure(figsize=(8, 6))
cp = plt.contour(T0, T1, J_vals, levels=np.logspace(-2, 3, 30), cmap='jet')
plt.clabel(cp, inline=1, fontsize=10)
plt.xlabel('θ₀')
plt.ylabel('θ₁')
plt.title('Contorno da Função Custo J(θ)')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2, label='θ encontrado')
plt.legend()
plt.grid(True)
plt.savefig("contorno_custo_theta.png")
plt.show()

#------------------------
# Métricas de Avaliação (Exercício seguinte)
#------------------------
# AUC, F1 Score e Matriz de Confusão não se aplicam diretamente à regressão,
# mas serão tratados em classificação (ex2data2) posteriormente.

final_cost = computeCost(X_prep, y, theta)
print("Theta final:", theta.ravel())
print(f"Custo final J(θ): {final_cost:.4f}")

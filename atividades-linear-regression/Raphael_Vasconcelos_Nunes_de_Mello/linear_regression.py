# Atividade 02 - Regressão linear
# Aluno: Raphael Vasconcelos Nunes de Mello
# Pc: Macbook Pro - M4, memória ram: 24 Gb.

import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():
    return np.eye(5)

print("Running warmUpExercise...")
print("5x5 Identity Matrix:")
print(warmUpExercise())

data = np.loadtxt('data/ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

def plotData(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Population size in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Profit vs Population')
    plt.show()

plotData(X, y)

def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    return 1 / (2 * m) * np.sum(sqrErrors)

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (X.T.dot(errors))
    return theta

X_with_intercept = np.column_stack((np.ones(m), X))
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

print("\nTesting the cost function...")
J = computeCost(X_with_intercept, y, theta)
print(f"With theta = [0 ; 0]\nCost computed = {J}")
print("Expected cost value (approx) 32.07")

J = computeCost(X_with_intercept, y, np.array([-1, 2]))
print(f"With theta = [-1 ; 2]\nCost computed = {J}")
print("Expected cost value (approx) 54.24")

theta = gradientDescent(X_with_intercept, y, theta, alpha, iterations)
print("Theta found by gradient descent:")
print(theta)
print("Expected theta values (approx): [-3.6303, 1.1664]")

plt.scatter(X, y, marker='x', c='r', label='Training data')
plt.plot(X, X_with_intercept @ theta, '-', label='Linear regression')
plt.xlabel('Population size in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()
plt.show()

predict1 = np.array([1, 3.5]) @ theta
predict2 = np.array([1, 7]) @ theta
print(f"For population = 35,000, we predict a profit of {predict1 * 10000}")
print(f"For population = 70,000, we predict a profit of {predict2 * 10000}")

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        t = np.array([theta0, theta1])
        J_vals[i, j] = computeCost(X_with_intercept, y, t)

J_vals = J_vals.T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(T0, T1, J_vals, cmap='viridis')
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('Cost J')
plt.show()

plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show()

data = np.loadtxt('data/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = len(y)

print("First 10 examples from the dataset:")
for i in range(10):
    print(f"x = [{X[i, 0]:.0f}, {X[i, 1]:.0f}], y = {y[i]:.0f}")

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

print("Normalizing Features...")
X, mu, sigma = featureNormalize(X)

X = np.column_stack((np.ones(m), X))

def computeCostMulti(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    return 1 / (2 * m) * np.sum(sqrErrors)

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * (X.T.dot(errors))
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

alpha = 0.01
num_iters = 800

theta = np.zeros(X.shape[1])
print("Running gradient descent...")

theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.plot(range(1, len(J_history) + 1), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of Gradient Descent')
plt.show()

print("Theta computed from gradient descent:")
print(theta)

normalized_features_predict = (np.array([1650, 3]) - mu) / sigma
normalized_features_predict = np.insert(normalized_features_predict, 0, 1)
price = normalized_features_predict @ theta

print(f"Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $ {price:.2f}")
print(f'Expected price (approx): 289314.62')

alpha_values = [1., 0.3, 0.1, 0.03, 0.01]
num_iters_experiment = 50

for alpha_exp in alpha_values:
    theta_exp = np.zeros(X.shape[1])
    theta_exp, J_history_exp = gradientDescentMulti(X, y, theta_exp, alpha_exp, num_iters_experiment)
    plt.plot(range(1, len(J_history_exp) + 1), J_history_exp, label=f'alpha = {alpha_exp}')

plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of Gradient Descent for different learning rates')
plt.legend()
plt.show()

print("Theta computed from gradient descent (last experiment shown):")
print(theta_exp)


def normalEqn(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta

print("Solving with normal equations...")
data_neq = np.loadtxt('data/ex1data2.txt', delimiter=',')
X_neq = data_neq[:, :2]
y_neq = data_neq[:, 2]

X_neq = np.column_stack((np.ones(len(y_neq)), X_neq))

theta_neq = normalEqn(X_neq, y_neq)

print("Theta computed from the normal equations:")
print(theta_neq)

price_neq = np.array([1, 1650, 3]) @ theta_neq
print(f"Predicted price of a 1650 sq-ft, 3 br house (using normal equations): $ {price_neq:.2f}")
print(f'Expected price (approx): 293081.46')
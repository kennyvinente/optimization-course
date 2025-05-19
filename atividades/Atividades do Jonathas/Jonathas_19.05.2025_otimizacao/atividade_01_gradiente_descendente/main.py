#------------------------
# Import packages
#------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Change the plotting style
plt.style.use('seaborn-v0_8-whitegrid')

#------------------------
# Create functions
#------------------------
# Input into the functions is vector X, where X = (x,y)
# Create the function we want to find the minimum of
def f(X):
    """Objective function: f(x,y) = (x-47)^2 + (y-0.1)^2 + 2"""
    x, y = X
    return (x - 47)**2 + (y - 0.1)**2 + 2

# Create the gradient vector
def gradf(X):
    """Gradient of f: ∇f(x,y) = [2(x-47), 2(y-0.1)]"""
    x, y = X
    return np.array([2*(x - 47), 2*(y - 0.1)])

#------------------------
# Plot the function
#------------------------
def plot_function_3d():
    """Plot the 3D surface of the function"""
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    x_range = np.linspace(45, 49, 50)
    y_range = np.linspace(-2, 2, 50)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    Z_mesh = f([X_mesh, Y_mesh])
    
    ax1.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D Surface Plot of f(x,y)')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    levels = np.logspace(0.3, 3, 20)
    contour = ax2.contour(X_mesh, Y_mesh, Z_mesh, levels=levels)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot of f(x,y)')
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.show()

# Plot the function
plot_function_3d()

#------------------------
# Initial gradient descent implementation
#------------------------
def simple_gradient_descent():
    """Simple gradient descent starting at (50, 2)"""
    x_current = np.array([50.0, 2.0])
    alpha = 0.5  # From the optimal step size calculation
    max_iterations = 100
    tolerance = 1e-6
    
    path = [x_current.copy()]
    f_values = [f(x_current)]
    
    for i in range(max_iterations):
        gradient = gradf(x_current)
        x_new = x_current - alpha * gradient
        
        # Check convergence
        if np.linalg.norm(x_new - x_current) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
            
        x_current = x_new
        path.append(x_current.copy())
        f_values.append(f(x_current))
    
    return np.array(path), np.array(f_values)

# Run simple gradient descent
path_simple, f_values_simple = simple_gradient_descent()
print(f"Final point: {path_simple[-1]}")
print(f"Final function value: {f_values_simple[-1]}")

#------------------------
# Feature normalization function
#------------------------
def feature_normalize(X):
    """
    Normalize features using mean and standard deviation
    
    Parameters:
    -----------
    X : numpy array
        Input data to normalize
        
    Returns:
    --------
    X_norm : numpy array
        Normalized data
    mu : numpy array
        Mean of each feature
    sigma : numpy array
        Standard deviation of each feature
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

#------------------------
# Advanced gradient descent implementation
#------------------------
def gradient_descent(f, grad_f, x0, alpha_type='fixed', alpha=0.01, 
                     max_steps=1000, tolerance=0.0001, 
                     c1=1e-4, rho=0.5, max_line_search_iter=20):
    """
    Gradient descent optimization algorithm.
    
    Parameters:
    -----------
    f : function
        The objective function to minimize, should take a numpy array as input.
    grad_f : function
        The gradient of the objective function, should take a numpy array as input.
    x0 : numpy array
        The starting point.
    alpha_type : str, optional
        Type of step size: 'fixed' or 'backtracking'. Default is 'fixed'.
    alpha : float, optional
        Step size for fixed alpha. Default is 0.01.
    max_steps : int, optional
        Maximum number of iterations. Default is 1000.
    tolerance : float, optional
        Convergence tolerance based on the norm of the gradient. Default is 0.0001.
    c1 : float, optional
        Parameter for the Armijo condition in backtracking line search. Default is 1e-4.
    rho : float, optional
        Step size reduction factor for backtracking line search. Default is 0.5.
    max_line_search_iter : int, optional
        Maximum number of backtracking iterations. Default is 20.
        
    Returns:
    --------
    X : numpy array
        The final solution point.
    f_values : list
        Function values at each iteration.
    path : list of numpy arrays
        The path taken by the algorithm.
    num_steps : int
        Number of steps taken to converge.
    gradient_norms : list
        Norm of gradient at each iteration
    """
    x_current = np.array(x0, dtype=float)
    path = [x_current.copy()]
    f_values = [f(x_current)]
    gradient_norms = []
    
    for step in range(max_steps):
        gradient = grad_f(x_current)
        grad_norm = np.linalg.norm(gradient)
        gradient_norms.append(grad_norm)
        
        # Check convergence based on gradient norm
        if grad_norm < tolerance:
            print(f"Converged after {step} iterations (gradient norm: {grad_norm:.2e})")
            break
        
        # Determine step size
        if alpha_type == 'fixed':
            step_size = alpha
        elif alpha_type == 'backtracking':
            step_size = backtracking_line_search(f, grad_f, x_current, gradient, 
                                               alpha, c1, rho, max_line_search_iter)
        else:
            raise ValueError("alpha_type must be 'fixed' or 'backtracking'")
        
        # Update position
        x_current = x_current - step_size * gradient
        path.append(x_current.copy())
        f_values.append(f(x_current))
    
    return x_current, f_values, path, step + 1, gradient_norms

def backtracking_line_search(f, grad_f, x, gradient, alpha_init, c1, rho, max_iter):
    """
    Backtracking line search using Armijo condition
    
    Parameters:
    -----------
    f : function
        Objective function
    grad_f : function
        Gradient function
    x : numpy array
        Current point
    gradient : numpy array
        Current gradient
    alpha_init : float
        Initial step size
    c1 : float
        Armijo condition parameter
    rho : float
        Step size reduction factor
    max_iter : int
        Maximum backtracking iterations
        
    Returns:
    --------
    alpha : float
        Step size satisfying Armijo condition
    """
    alpha = alpha_init
    f_x = f(x)
    grad_norm_sq = np.dot(gradient, gradient)
    
    for _ in range(max_iter):
        x_new = x - alpha * gradient
        f_new = f(x_new)
        
        # Armijo condition
        if f_new <= f_x - c1 * alpha * grad_norm_sq:
            return alpha
        
        alpha *= rho
    
    return alpha

#------------------------
# Test functions
#------------------------
def f1(X):
    """Function 1: f(x,y) = x^2 + y^2"""
    x, y = X
    return x**2 + y**2

def grad_f1(X):
    """Gradient of function 1"""
    x, y = X
    return np.array([2*x, 2*y])

def f2(X):
    """Function 2: f(x,y) = (x-1)^2 + (y-1)^2"""
    x, y = X
    return (x-1)**2 + (y-1)**2

def grad_f2(X):
    """Gradient of function 2"""
    x, y = X
    return np.array([2*(x-1), 2*(y-1)])

def f3(X):
    """Function 3: Rosenbrock function f(x,y) = 20(y-x^2)^2 + (1-x)^2"""
    x, y = X
    return 20*(y - x**2)**2 + (1 - x)**2

def grad_f3(X):
    """Gradient of Rosenbrock function"""
    x, y = X
    dx = -80*x*(y - x**2) - 2*(1 - x)
    dy = 40*(y - x**2)
    return np.array([dx, dy])

#------------------------
# Visualization functions
#------------------------
def plot_convergence_path(func, path, title, x_range=None, y_range=None, levels=20):
    """
    Plot the convergence path on a contour plot
    
    Parameters:
    -----------
    func : function
        The objective function
    path : list of numpy arrays
        The optimization path
    title : str
        Plot title
    x_range : list, optional
        X-axis range [min, max]
    y_range : list, optional
        Y-axis range [min, max]
    levels : int, optional
        Number of contour levels
    """
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
    
    # Plot optimization path
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=8, label='Optimization path')
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start point')
    plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End point')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence_analysis(f_values, gradient_norms, title):
    """
    Plot convergence analysis: function values and gradient norms
    
    Parameters:
    -----------
    f_values : list
        Function values at each iteration
    gradient_norms : list
        Gradient norms at each iteration
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Function values
    ax1.plot(f_values, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function Value')
    ax1.set_title(f'{title} - Function Values')
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Gradient norms
    ax2.plot(gradient_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title(f'{title} - Gradient Norms')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

#------------------------
# Performance metrics for classification (if applicable)
#------------------------
def calculate_f1_score(y_true, y_pred):
    """Calculate F1 score for binary classification"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    from collections import Counter
    
    # Calculate confusion matrix
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#------------------------
# Test cases
#------------------------
def run_comprehensive_tests():
    """Run comprehensive tests for different functions and parameters"""
    
    # Test functions and their gradients
    functions = [
        (f1, grad_f1, "f(x,y) = x² + y²"),
        (f2, grad_f2, "f(x,y) = (x-1)² + (y-1)²"),
        (f3, grad_f3, "Rosenbrock function")
    ]
    
    # Different starting points
    starting_points = [
        [0, 0],
        [5, 5],
        [-2, -2],
        [10, -5]
    ]
    
    # Different alpha values for fixed step size
    alpha_values = [0.001, 0.01, 0.1, 0.5]
    
    print("=== COMPREHENSIVE GRADIENT DESCENT TESTS ===\n")
    
    # Test 1: Different alpha values for each function
    for func, grad_func, func_name in functions:
        print(f"\nTesting {func_name}")
        print("-" * 50)
        
        for alpha in alpha_values:
            result = gradient_descent(func, grad_func, [0, 0], 
                                    alpha_type='fixed', alpha=alpha, 
                                    max_steps=1000, tolerance=1e-6)
            
            final_point, f_values, path, num_steps, grad_norms = result
            print(f"Alpha={alpha:5.3f}: Steps={num_steps:3d}, Final f={f_values[-1]:8.6f}, "
                  f"Final point=({final_point[0]:6.3f}, {final_point[1]:6.3f})")
    
    # Test 2: Backtracking line search vs fixed alpha
    print("\n\n=== BACKTRACKING LINE SEARCH COMPARISON ===\n")
    
    for func, grad_func, func_name in functions:
        print(f"\nComparing methods for {func_name}")
        print("-" * 50)
        
        # Fixed alpha
        result_fixed = gradient_descent(func, grad_func, [0, 0], 
                                       alpha_type='fixed', alpha=0.01, 
                                       max_steps=1000, tolerance=1e-6)
        
        # Backtracking
        result_backtrack = gradient_descent(func, grad_func, [0, 0], 
                                          alpha_type='backtracking', alpha=1.0, 
                                          max_steps=1000, tolerance=1e-6)
        
        print(f"Fixed alpha    : Steps={result_fixed[3]:3d}, Final f={result_fixed[1][-1]:8.6f}")
        print(f"Backtracking   : Steps={result_backtrack[3]:3d}, Final f={result_backtrack[1][-1]:8.6f}")
        
        # Plot comparison
        plot_convergence_path(func, result_fixed[2], 
                            f"{func_name} - Fixed Alpha (α=0.01)", 
                            x_range=[-3, 3], y_range=[-3, 3])
        
        plot_convergence_path(func, result_backtrack[2], 
                            f"{func_name} - Backtracking Line Search", 
                            x_range=[-3, 3], y_range=[-3, 3])
        
        # Plot convergence analysis
        plot_convergence_analysis(result_fixed[1], result_fixed[4], 
                                f"{func_name} - Fixed Alpha")
        
        plot_convergence_analysis(result_backtrack[1], result_backtrack[4], 
                                f"{func_name} - Backtracking")

# Run all tests
run_comprehensive_tests()

#------------------------
# Special test for Rosenbrock function (from the code snippet)
#------------------------
def test_rosenbrock_detailed():
    """Detailed test for Rosenbrock function as shown in the original code"""
    
    print("\n=== DETAILED ROSENBROCK FUNCTION TEST ===\n")
    
    # Test with fixed alpha
    result_fixed = gradient_descent(f3, grad_f3, [0, 0], 
                                   alpha_type='fixed', alpha=0.01, 
                                   max_steps=10000, tolerance=1e-8)
    
    plot_convergence_path(f3, result_fixed[2], 
                         "Gradient Descent with Fixed Alpha=0.01", 
                         x_range=[-2, 2], y_range=[-2, 2])
    
    # Test with backtracking
    result_backtrack = gradient_descent(f3, grad_f3, [0, 0], 
                                       alpha_type='backtracking', alpha=1.0, 
                                       max_steps=10000, tolerance=1e-8)
    
    plot_convergence_path(f3, result_backtrack[2], 
                         "Gradient Descent with Backtracking", 
                         x_range=[-2, 2], y_range=[-2, 2])
    
    # Print detailed results
    print(f"Fixed Alpha Results:")
    print(f"  Final point: ({result_fixed[0][0]:.6f}, {result_fixed[0][1]:.6f})")
    print(f"  Final function value: {result_fixed[1][-1]:.8f}")
    print(f"  Number of iterations: {result_fixed[3]}")
    print(f"  Final gradient norm: {result_fixed[4][-1]:.2e}")
    
    print(f"\nBacktracking Results:")
    print(f"  Final point: ({result_backtrack[0][0]:.6f}, {result_backtrack[0][1]:.6f})")
    print(f"  Final function value: {result_backtrack[1][-1]:.8f}")
    print(f"  Number of iterations: {result_backtrack[3]}")
    print(f"  Final gradient norm: {result_backtrack[4][-1]:.2e}")

# Run detailed Rosenbrock test
test_rosenbrock_detailed()

print("\n=== GRADIENT DESCENT ANALYSIS COMPLETE ===")
print("The implementation includes:")
print("- Fixed and adaptive (backtracking) step sizes")
print("- Armijo condition for line search")
print("- Convergence monitoring via gradient norm")
print("- Comprehensive visualization of optimization paths")
print("- Performance analysis for multiple test functions")
print("- Feature normalization capabilities")
print("- All implemented using only NumPy and Matplotlib")
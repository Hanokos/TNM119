import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

##### Load the Data #####
with open('Task_4_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Convert to Pandas DataFrame (if not already)
data = pd.DataFrame(data)

# X (features) and y (target)
X = data.iloc[:, :-1].to_numpy()  # All columns except last
y = data.iloc[:, -1].to_numpy()   # Last column (target)

# Add a bias term to X
X = np.c_[np.ones(X.shape[0]), X]  # Adds a column of 1s for bias

##### Gradient Descent for Linear Regression #####
def gradient_descent(X, y, learning_rate=0.1, max_iterations=1000, tolerance=1e-5):
    """
    Performs gradient descent to minimize Mean Squared Error (MSE).
    """
    m, n = X.shape  
    weights = np.zeros(n)  # Initialize weights
    cost_history = []

    for i in range(max_iterations):
        predictions = X @ weights  
        error = predictions - y
        cost = np.mean(error ** 2)  
        cost_history.append(cost)

        gradient = (2 / m) * (X.T @ error)  # Compute gradient
        weights -= learning_rate * gradient  # Update weights

        # Check for convergence
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
            return weights, cost_history, i + 1   # Stop early if improvement is too small

    return weights, cost_history, max_iterations  

# Run gradient descent
optimized_weights, cost_history, num_iterations = gradient_descent(X, y)

# Print results
print(f"Converged in {num_iterations} iterations with final cost = {cost_history[-1]:.6f}")

##### Display the Learned Linear Regression Model #####
print("\nLearned Linear Regression Model:")
equation = "y = " + " + ".join([f"{w:.4f}*x{i}" for i, w in enumerate(optimized_weights)])
print(equation)

##### Plot Cost Function Convergence #####
plt.plot(range(num_iterations), cost_history, label='Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Convergence')
plt.legend()
plt.grid()
plt.show()

##### Plot Regression Line (Only for 1 Feature) #####
if X.shape[1] == 2:  # 1 feature + bias term
    plt.scatter(X[:, 1], y, label='Actual Data', color='blue')
    plt.plot(X[:, 1], X @ optimized_weights, label='Regression Line', color='red')
    plt.xlabel('Feature X')
    plt.ylabel('Target y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()

##### Compare Actual vs. Predicted Values #####
predictions = X @ optimized_weights
plt.scatter(y, predictions, color='green')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle="--", color="red")  # Perfect fit line
plt.show()




# ÄR NÄSTAN KORREKT: denna kod fattar varibelb "b vilket är samma som w0".
#  DEtta anvädnas i formeln f(x)= b + w1*x1 + wn*xn .... 
# där b är värdet för funktionen upp o neråt i y led där funktionen startar
# w är vikt som är okänd och räknas ut med modellen
# x är input
# f(x) = y är output som är dom  justerade vikterna "optimized_weights"
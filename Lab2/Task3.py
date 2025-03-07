import numpy as np
import pickle
import pandas as pd  # Import pandas to handle DataFrame case
import matplotlib.pyplot as plt  # Import matplotlib for visualization

#READ  to understand my code: 
# This code's purpose is to measure how far the predictions are from the actual values.
# Goal: Minimize this cost!

##### Load the data here #####
with open('Task_3_data.pkl', 'rb') as file: # "rb" means "r = read" and "b = binary"
    data = pickle.load(file) # using pickle to load in data

# Ensure compatibility whether data is a DataFrame or NumPy array
if isinstance(data, pd.DataFrame): # check if data is a DataFrame
    num_features = data.shape[1] - 1  # Last column is treated as y
    X = data.iloc[:, :num_features].values  # select x as all columns, except for the last column
    y = data.iloc[:, -1].values  # select the last column to be y
    # ".values" = Convert to NumPy array
else:
    num_features = data.shape[1] - 1
    X = np.array(data[:, :num_features])
    y = np.array(data[:, -1])

##### Convert the data into numpy arrays here ########
X = np.array(X)
y = np.array(y)

########## End of Data preparation ##############

learning_rate = 0.6 # Controls how big the updates are
# Too high: The optimizer jumps around and never converging
# Too low: The optimizer moves too slowly, taking long to reduce cost

# Cost function,(MSE)
def cost_function(X, y, weights):
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)

# Gradient of the cost function, (adjust the weights)
def cost_function_gd(X, y, weights):
    predictions = X @ weights
    return (2 / len(y)) * X.T @ (predictions - y)

# Optimizer 1: Adam Optimizer
def adam_optimizer(X, y, init_weights, iterations):
    weights = init_weights.copy()
    beta1, beta2 = 0.8, 0.99 # changed this value to make it pass final cost for question 3
    epsilon = 1e-8
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    costs = []
    for t in range(1, iterations + 1):
        grad = cost_function_gd(X, y, weights)
        m = beta1 * m + (1 - beta1) * grad # momentum
        v = beta2 * v + (1 - beta2) * (grad ** 2) 
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) # adjust learning rate
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Optimizer 2: RMSprop
def rmsprop_optimizer(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    decay_rate = 0.9 # changed it to 0.9 to get to 0.09 as final cost
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum = decay_rate * grad_accum + (1 - decay_rate) * grad ** 2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon) # adjust the learning rate 
        weights -= adjusted_lr * grad # adjust after each step
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Optimizer 3: Momentum-based Gradient Descent
def momentum_gd_optimizer(X, y, init_weights, iterations):
    weights = init_weights.copy()
    v = np.zeros_like(weights) 
    momentum = 0.1
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        v = momentum * v - learning_rate * grad # store past momentum (gradient) and update
        weights += v # remember past momentum (gradient)
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Optimizer 4: AdaGrad
def adagrad_optimizer(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum += grad ** 2 # sums up past gradient
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon) # adjust learning rate
        weights -= adjusted_lr * grad # the steps get smaller, dvs adjusting itself
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Optimizer 5: Standard Gradient Descent
def standard_gd_optimizer(X, y, init_weights, iterations):
    weights = init_weights.copy()
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        weights -= learning_rate * grad # no momentum or adjusting learning rate, meaning equal step size
        costs.append(cost_function(X, y, weights))
    return weights, costs

########## OWN MADE CODE: to produce an output ##############

init_weights = np.zeros(num_features)
iterations = 20

optimizers = [
    ("Adam", adam_optimizer),
    ("RMSprop", rmsprop_optimizer),
    ("Momentum GD", momentum_gd_optimizer),
    ("AdaGrad", adagrad_optimizer),
    ("Standard GD", standard_gd_optimizer)
]

optimizer_costs = {}

for name, optimizer in optimizers:
    weights, costs = optimizer(X, y, init_weights, iterations)
    if costs[-1] > 1e-3:  # If not converging, adjust learning rate and retry
        learning_rate = 0.5
        weights, costs = optimizer(X, y, init_weights, iterations)
    optimizer_costs[name] = costs
    print(f"{name}: Final Cost after {iterations} iterations = {costs[-1]}")

# Plot the convergence of each optimizer
plt.figure(figsize=(10, 6))
for name, costs in optimizer_costs.items():
    plt.plot(range(1, iterations + 1), costs, label=name)

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Convergence of Different Optimizers")
plt.legend()
plt.yscale("log")  # Use log scale to better visualize differences
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create a meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Updated membership functions with specific parameters
def mu_small_x(x):
    return 1 / (1 + np.exp(5 * (x - (-5))))  # Sigmoid for "small x" centered at -5

def mu_large_x(x):
    return 1 / (1 + np.exp(-5 * (x - 5)))    # Sigmoid for "large x" centered at 5

def mu_small_y(y):
    return 1 / (1 + np.exp(2 * (y - (-2))))  # Sigmoid for "small y" centered at -2

def mu_large_y(y):
    return 1 / (1 + np.exp(-2 * (y - 2)))    # Sigmoid for "large y" centered at 2

# Rule functions
def rule1(x, y):
    return -x + y + 1

def rule2(x, y):
    return -y + 3

def rule3(x, y):
    return -x + 3

def rule4(x, y):
    return x + y + 2

# Evaluate the contribution of each rule based on membership values
def evaluate_rules(X, Y):
    mu_x_small = mu_small_x(X)
    mu_x_large = mu_large_x(X)
    mu_y_small = mu_small_y(Y)
    mu_y_large = mu_large_y(Y)
    
    R1 = mu_x_small * mu_y_small * rule1(X, Y)
    R2 = mu_x_small * mu_y_large * rule2(X, Y)
    R3 = mu_x_large * mu_y_small * rule3(X, Y)
    R4 = mu_x_large * mu_y_large * rule4(X, Y)
    
    output = (R1 + R2 + R3 + R4) / (mu_x_small * mu_y_small + mu_x_small * mu_y_large + mu_x_large * mu_y_small + mu_x_large * mu_y_large)
    return output

# Calculate new output z with the updated membership functions
Z_new = evaluate_rules(X, Y)

# Plotting the membership functions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(x, mu_small_x(x), label='Small x')
ax1.plot(x, mu_large_x(x), label='Large x')
ax1.set_title('Membership Functions for x')
ax1.set_ylabel('Membership value')
ax1.legend()

ax2.plot(y, mu_small_y(y), label='Small y')
ax2.plot(y, mu_large_y(y), label='Large y')
ax2.set_title('Membership Functions for y')
ax2.set_ylabel('Membership value')
ax2.legend()

plt.show()

# Plot the new surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_new, cmap='viridis')
plt.title('Output Surface Z of Takagi-Sugeno Model')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.colorbar(surf)
plt.show()

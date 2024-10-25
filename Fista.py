# Import necessary libraries
import time                # For timing the execution
from math import sqrt      # For mathematical computations
import numpy as np         # For numerical operations
import pandas as pd        # For data manipulation (not directly used here)

"""--------------------------------------FISTA Algorithm-----------------------------------------------------------------------"""

# Function for soft thresholding (l1 Norm regularization)
# Shrinks the values of x towards zero by l, a thresholding parameter
def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

# Function to compute the Lipschitz constant of the gradient for given data
# Used to ensure stable convergence in the FISTA algorithm
def compute_lipschitz_constant_of_gradient(data):
    # Compute the second derivative (numerical approximation) and find the maximum absolute value
    second_derivative = np.abs(np.gradient(np.gradient(data)))
    lipschitz_constant = np.max(second_derivative)
    return lipschitz_constant

# Main FISTA function for optimization
# Solves an optimization problem for given data, threshold (lambda) and maximum iterations
def fista(data, l, maxit):
    b = data                      # Target data for FISTA optimization
    x = np.zeros_like(data)       # Initialize x with zeros
    pobj = []                     # List to store objective function values over iterations
    t = 1                         # Initial acceleration parameter
    z = x.copy()                  # Initialize z for momentum-based updates
    L = compute_lipschitz_constant_of_gradient(data)  # Calculate Lipschitz constant
    
    time0 = time.time()           # Start timing the algorithm

    # FISTA iteration loop
    for _ in range(maxit):
        xold = x.copy()           # Store previous x value for momentum calculation
        
        # Update z based on the gradient step
        z = z + (b - z) / L
        
        # Apply soft thresholding to update x
        x = soft_thresh(z, l / L)
        
        # Update t and calculate momentum-based z for the next iteration
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        
        # Compute the objective function value (l2 loss + l1 regularization)
        this_pobj = 0.5 * np.linalg.norm(x - b) ** 2 + l * np.linalg.norm(x, 1)
        
        # Record time and objective function value
        pobj.append((time.time() - time0, this_pobj))

    # Convert pobj list of tuples to two separate arrays for time and objective values
    times, pobj = map(np.array, zip(*pobj))
    
    # Return the optimized x, objective values over time, and timing information
    return x, pobj, times

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Function to compute the exact solution of an underdamped harmonic oscillator
def exact_solution(d, w0, t):
    """
    Computes the exact analytical solution of an under-damped harmonic oscillator.
    
    Parameters:
        d  : float - Damping coefficient (must be smaller than natural frequency w0)
        w0 : float - Natural frequency of the system
        t  : tensor - Time values at which the solution is computed (PyTorch tensor)
    
    Returns:
        u : tensor - The displacement of the oscillator at time t
    """
    
    # Ensure that damping coefficient d is less than the natural frequency w0
    assert d < w0, "The system must be underdamped (d < w0) for oscillations to occur."

    # Compute the damped frequency of the oscillator
    w = np.sqrt(w0**2 - d**2)
    
    # Compute the phase shift based on damping
    phi = np.arctan(-d / w)

    # Compute the amplitude correction factor
    A = 1 / (2 * np.cos(phi))

    # Compute the oscillatory component
    cos = torch.cos(phi + w * t)

    # Compute the exponential decay component
    exp = torch.exp(-d * t)

    # Compute the final solution for displacement u
    u = exp * 2 * A * cos

    return u


# Custom activation function: Sine Activation
class SinActivation(nn.Module):
    """
    Defines a custom activation function using the sine function.
    This can be useful in certain types of neural networks, such as physics-informed neural networks (PINNs).
    """

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        """
        Forward pass for the activation function.

        Parameters:
            x : tensor - Input tensor
        
        Returns:
            torch.sin(x) : tensor - Element-wise sine activation applied to x
        """
        return torch.sin(x)


# Fully Connected Neural Network (FCN)
class FCN(nn.Module):
    """
    Defines a standard fully connected (dense) neural network in PyTorch.

    This network consists of:
    - An input layer followed by an activation function
    - Multiple hidden layers with activation functions
    - An output layer without activation

    Parameters:
        N_INPUT  : int - Number of input features
        N_OUTPUT : int - Number of output features
        N_HIDDEN : int - Number of neurons in each hidden layer
        N_LAYERS : int - Number of hidden layers
        activation : nn.Module - Activation function (default: Tanh)
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
        super().__init__()

        # Input layer: First fully connected layer with an activation function
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),  # Linear transformation
            activation()  # Activation function (e.g., Tanh)
        )

        # Hidden layers: Stack of fully connected layers with activation functions
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),  # Fully connected layer
                    activation()  # Activation function
                ) for _ in range(N_LAYERS - 1)  # Repeat for the number of hidden layers
            ]
        )

        # Output layer: Final linear transformation (no activation function)
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters:
            x : tensor - Input tensor
        
        Returns:
            x : tensor - Output tensor after passing through the network
        """
        
        x = self.fcs(x)  # Pass input through the first layer
        x = self.fch(x)  # Pass through the hidden layers
        x = self.fce(x)  # Final output layer (no activation)
        
        return x



import timeit as it
import numpy as np
import matplotlib.pyplot as plt
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim


observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

kwargs = {'observed': observed, 'predicted': predicted}

def sk_mse_interface(observed, predicted):
    return sk_mse(observed, predicted)

factory = {'mse_vanilla': vanilla_mse,
           'mse_numpy': numpy_mse,
           'mse_sk': sk_mse_interface
           }

for name, func in factory.items():
    exec_time = it.timeit('{func(**kwargs)}',
                          globals=globals(), number=100) / 100
    mse = func(**kwargs)
    print(f"Mean Squared Error, {name}: {mse}, "
          f"Average execution time: {exec_time} seconds")

def generate_data():
    '''
    Generate a sinusoidal function with noise
    
    Returns:
    x : numpy array
        The x values
    y_noisy : numpy array
        The noisy y values
    y_clean : numpy array
        The clean y values
    '''
    x = np.linspace(0, 10, 100)

    A, B, C = 1, 2, 0  
    y_clean = A * np.sin(B * x + C)
    
    noise = np.random.normal(scale=0.2, size=len(x))
    y_noisy = y_clean + noise

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_clean, label="Without noise")
    plt.plot(x, y_noisy, color='red', alpha=0.6, label="With noise")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Sinusoidal Function with Noise")
    plt.show()
    
    return x, y_noisy, y_clean

x, y_noisy, y_clean = generate_data()


def plot_kmeans(x, y_noisy):
    '''
    Plot the elbow method for KMeans
    
    Parameters:
    x : numpy array
        The x values
    y_noisy : numpy array
        The noisy y values
    '''
    cluster_range = range(1, 10)
    inertia_values = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(np.column_stack((x, y_noisy)))
        inertia_values.append(kmeans.inertia_)

    
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia_values, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Intra-cluster Inertia")
    plt.title("Variance vs. Number of Clusters")
    plt.show()

plot_kmeans(x, y_noisy)


def kmeans_clustering(x, y_noisy, k_optimal):

    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.column_stack((x, y_noisy)))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y_noisy, c=labels, cmap='viridis', alpha=0.6, label="Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"K-Means Clustering (k={k_optimal})")
    plt.legend()
    plt.show()

k_optimal = 4
kmeans_clustering(x, y_noisy, k_optimal)


def linear_regression(x, y_noisy):
    '''
    Perform linear regression on the noisy data
    
    Parameters:
    x : numpy array
        The x values
    y_noisy : numpy array
        The noisy y values
    '''
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y_noisy)
    y_pred_lr = lr.predict(x.reshape(-1, 1))
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y_noisy, label="Noisy Data", alpha=0.5)
    plt.plot(x, y_pred_lr, color="red", label="Linear Regression")
    plt.legend()
    plt.title("Linear Regression on Noisy Data")
    plt.show()

linear_regression(x, y_noisy)


X_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
Y_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 100000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, Y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

y_pred_nn = model(X_tensor).detach().numpy()
plt.figure(figsize=(8, 5))
plt.scatter(x, y_noisy, alpha=0.5, label="Noisy Data")
plt.plot(x, y_pred_nn, color="green", label="NN Prediction")
plt.legend()
plt.title("Regression with a Neural Network")
plt.show()

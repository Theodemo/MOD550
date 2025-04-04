import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Function to display examples from a dataset
def show_examples(dataset, num_examples=5):
    fig, axes = plt.subplots(1, num_examples, figsize=(12, 3))

    for i in range(num_examples):
        # Get a random example
        index = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[index]

        # Display the image
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.show()
    
# Define the MNIST datasets.
dataset_train_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
dataset_test_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)


# Display examples from the training set
print("Examples from the Training Set:")
show_examples(dataset_train_part)

# Display examples from the test set
print("\nExamples from the Test Set:")
show_examples(dataset_test_part)


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class SimpleConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(26 * 26 * 10, 50),
      nn.ReLU(),
      nn.Linear(50, 20),
      nn.ReLU(),
      nn.Linear(20, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

def train_model(network, trainloader, optimizer, loss_function, num_epochs):
    # Function to perform the training.
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        current_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            if i % 500 == 499:
                print(
                    'Loss after mini-batch %5d: %.3f' %
                     (i + 1, current_loss / 500)
                )
                current_loss = 0.0


def test_model(network, testloader):
   # Function to test the model on the test set for a fold
    correct, total = 0, 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


def k_fold_cross_validation(k_folds, num_epochs, loss_function):
    # Function to perform k-fold cross-validation
    results = {}
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        trainloader = DataLoader(
            dataset, batch_size=10, sampler=train_subsampler)
        testloader = DataLoader(
            dataset, batch_size=10, sampler=test_subsampler)

        network = SimpleConvNet()
        network.apply(reset_weights)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        train_model(network, trainloader, optimizer, loss_function, num_epochs)

        print('Training process has finished. Saving trained model.')
        print('Starting testing')

        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)

        accuracy = test_model(network, testloader)
        results[fold] = accuracy
        print('--------------------------------')

    return results

# Function to print k-fold cross-validation results
def print_results(results):
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {len(results)} FOLDS')
    print('--------------------------------')

    total_accuracy = sum(results.values())
    average_accuracy = total_accuracy / len(results)

    for key, value in results.items():
        print(f'Fold {key}: {value:.2f}%')

    print(f'Average: {average_accuracy:.2f}%')

# Main function
def main():
    k_folds = 5
    num_epochs = 1
    loss_function = nn.CrossEntropyLoss()

    results = k_fold_cross_validation(k_folds, num_epochs, loss_function)
    print_results(results)
    
main()
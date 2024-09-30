from sklearn.datasets import fetch_openml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------Data Preprocessing-----------------------#
# Load the MNIST dataset
def load_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser="auto")
    X, y = torch.from_numpy(X.to_numpy()).float(), torch.from_numpy(y.to_numpy().astype(np.uint8))
    print(X.shape, y.shape, X.dtype, y.dtype)
    return X, y

# Split the dataset
def train_test_split(raw_data, labels, split_index):
	"""Split the original data into a training dataset
	and a testing dataset.

	Args:
		raw_data: An array of shape [n_samples, 784].
        labels : An array of shape [n_samples,].
		split_index: An integer.
    Returns:
        X_train, X_test, y_train, y_test(follow this order)

	"""
    ### YOUR CODE HERE

	### END YOUR CODE


def prepare_X(raw_X):
    """Normalize raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 784].

    Returns:
        X: Normalized samples [n_samples, 784].
    """
	### YOUR CODE HERE

	### END YOUR CODE

# --------------Model------------------#
#A simple linear MLR model
class MLR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifer = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.classifer(x)

# --------------------Main function----------------------#
def main():
    # Hyperparameters
    input_size = 28 * 28  # 784, for flattened 28x28 MNIST images
    num_classes = 10  # Digits 0-9
    learning_rate = 0.01
    batch_size = 1
    num_epochs = 20
    weight_decay = 0.0
    eval_freq = 5000/batch_size
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the MNIST dataset
    X, y = load_data()
    print()
    # Split data into training, validation and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 60000) #first split original data into trainset and testset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 50000) #then split trainset into trainset and valset

    # Preprocess raw data
    X_train = prepare_X(X_train)
    X_val = prepare_X(X_val)
    X_test = prepare_X(X_test)

    # Create the Datasets
    trainset = TensorDataset(X_train, y_train)
    valset = TensorDataset(X_val, y_val)
    testset = TensorDataset(X_test, y_test)
    # Create the Dataloaders
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    train_loader_eval = DataLoader(dataset=trainset, batch_size=1000, shuffle=False) #for evaluation, set large batchsize
    val_loader = DataLoader(dataset=valset, batch_size=1000, shuffle=False) #batchsize for validate set and test set doesn't affect
    test_loader = DataLoader(dataset=testset, batch_size=1000, shuffle=False) #training, set large batchsize to facilitate parallelism

    # Initialize model, loss function, and optimizer
    ### YOUR CODE HERE

	### END YOUR CODE

    # Training loop and result printing
    ### YOUR CODE HERE

	### END YOUR CODE

main()

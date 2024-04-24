import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Dataset object
class CustomEEGDataset(Dataset):
    def __init__(self, annotations_file, eeg_file, transform=None, target_transform=None):
        self.eeg_labels = torch.from_numpy(np.load(annotations_file).reshape(-1,1))
        self.eeg_data = torch.from_numpy(np.load(eeg_file))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.eeg_labels)

    def __getitem__(self, idx):
        label = self.eeg_labels[idx]
        eeg = self.eeg_data[idx]
        if self.transform:
            eeg = self.transform(eeg)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg, label
    
# Model definition 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32,2)
        self.fc1 = nn.Linear(896, 120)
        self.fc2 = nn.Linear(120, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
# training function 
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
# instantiation
eeg_DE_dataset = CustomEEGDataset('/data/label_valence.npy','/data/eeg_data.npy')



########################################## Training ##########################################
# Define the number of folds and batch size
k_folds = 5
batch_size = 32

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True)

# Loop through each fold
for fold, (train_idx, test_idx) in enumerate(kf.split(eeg_DE_dataset)):
    print(f"Fold {fold + 1}")
    print("-------")

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=eeg_DE_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )

    test_loader = DataLoader(
        dataset=eeg_DE_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    # Initialize the model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model on the current fold
    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer, epoch)
    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    correct = 0
    sample_eval = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.binary_cross_entropy(output, target).item()
            pred = output
            correct += int(sum(pred.eq(target))[0])
            sample_eval += int(target.shape[0])

    test_loss /= sample_eval
    accuracy = 100.0 * correct / sample_eval

    #Print the results for the current fold
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{sample_eval} ({accuracy:.2f}%)\n")

output_dict = {'test_loss': test_loss, 'Accuracy': correct/sample_eval}
output_df = pd.DataFrame.from_dict(output_dict)
output_df.to_csv('results1')
torch.save(model.state_dict(),'dumb_cnn.pt')
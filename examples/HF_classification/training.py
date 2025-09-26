import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
from typing import Tuple
import random

import torch
import torchvision

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import os


class DataManger:
    """
    Manages data by loading and splitting data into training and test sets.
    It also consists of functions to help navigate the dataset.

    Note: Singleton Pattern. Singleton pattern ensures 
    a single shared instance of DataManger per agent.

    Attributes:
            data (pd.DataFrame): The loaded dataset from the CSV file. 
            x (torch.Tensor): Feature matrix from the loaded dataset.
            y (torch.Tensor): Labels from the loaded dataset. 
            trainloader (DataLoader): DataLoader for the training set.
            testloader (DataLoader): DataLoader for the test set.
            test_s (int): Number of samples in the test set.
            cutoff_threshold (int): Limit on how many samples to use 
            in training (for simulating limited data).
    """

    _singleton_dm = None

    @classmethod
    def dm(cls, agent_nr: str, th: int = 0) -> "DataManger":
        if not cls._singleton_dm and th > 0:
            cls._singleton_dm = cls(agent_nr,th)

        return cls._singleton_dm

    def __init__(self, agent_nr: str,cutoff_th: int) -> None:
        

        TRAIN_SPLIT = 0.8
        BATCH_SIZE_TRAIN = 16
        BATCH_SIZE_TEST = 8
        NUM_WORKERS = 2

        RANDOM_STATE = 42
        
        self.set_seed(RANDOM_STATE)

        # Load dataset from CSV and separate features and labels
        data_paths = {
    'CL': "examples/HF_classification/data/CL/whole_data.csv",
    'a1': "examples/HF_classification/data/clients/client1.csv",
    'a2': "examples/HF_classification/data/clients/client2.csv",
        }

        try:
            csv_file = data_paths[agent_nr]
        except KeyError:
            raise ValueError("agent_nr must be one of: 'CL', 'a1', or 'a2'")
        
        logging.info(f'data derived from {csv_file}')
        self.data = pd.read_csv(csv_file)

        # Wrap features and labels into a PyTorch dataset object
        x = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        full_data = torch.utils.data.TensorDataset(self.x, self.y)
        train_size = int(TRAIN_SPLIT * len(full_data))
        test_size = len(full_data) - train_size
        self.test_s = test_size
        train_data, test_data = random_split(full_data, [train_size, test_size])

        # Create data loaders for training and test sets
        self.trainloader = DataLoader(train_data, batch_size = BATCH_SIZE_TRAIN, 
        shuffle=True, num_workers=NUM_WORKERS) 
        self.testloader = DataLoader(test_data, batch_size = BATCH_SIZE_TEST, 
        shuffle=False, num_workers=NUM_WORKERS)

        self.cutoff_threshold = cutoff_th

    def __len__(self) -> int:
        """
        Return the total number of samples in the datasets.

        Returns:
            int
        """

        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get access to a specific sample by index, returning features and its 
        corresponding label. 

        Args:
            index (int): index of the specific sample.

        Returns:
            features, labels (Tuple[torch.Tensor, torch.Tensor]). 
        """

        features = torch.tensor(self.x[index], dtype=torch.float32)
        label = torch.tensor(self.y[index], dtype=torch.long)

        return features, label


    def get_random_samples(self, is_train: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of input samples and their corresponding labels.
        Useful for showing examples for demos.

        Args:
            is_train (bool): Whether to sample from training data.
        
        Returns:
            samples, labels (Tuple[torch.Tensor, torch.Tensor]). 
        """

        if is_train:
            ldr = self.trainloader
        else:
            ldr = self.testloader
        samples, labels = iter(ldr).next()

        return samples, labels


    def set_seed(self, seed: int =42) -> None:
        """
        Set a random seed for reproducibility.

        Args:
            seed (int): The random seed to use.
        
        Returns:
            None
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def execute_th_training(
    dm: DataManger, 
    net: torch.nn.Module, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer) -> torch.nn.Module:
    """
    Trains a neural network using only a sampled subset of the data, 
    based on the cutoff threshold

    Args:
        dm (DataManger): Data manager providing loaders and cutoff.
        net (torch.nn.Module): Neural network model.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        torch.nn.Module: The trained model.
    """ 
    BINARY = True
    EPOCH_NR = 5

    num_samples = len(dm.trainloader.dataset)

    # Ensure cutoff_threshold does not exceed the number of samples
    sample_size = min(dm.cutoff_threshold, num_samples)
    
    # Randomly sample a fixed number number of indices
    # to simulate limited local training data
    random_indices = random.sample(range(0, num_samples), sample_size)

    for epoch in range(EPOCH_NR):
        running_loss = 0.0
        j = 0
        for i, data in enumerate(dm.trainloader):
            # Only use a sampled subset of batches to 
            # simulate limited client data
            if i in random_indices:
                j += 1

                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                ## Perform forward pass, compute loss, 
                # backward pass, and optimizer step
                outputs = net(inputs)
    
                # Convert labels to float for 
                # binary classification using BCEWithLogitsLoss
                if BINARY:
                    labels = labels.float().unsqueeze(1) 
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # log average running loss every 1000 mini-batches
                running_loss += loss.item()
                if j % 1000 == 999:  
                    logging.info('[%d, %5d] loss: %.3f' %
                          (epoch + 1, j + 1, running_loss / 1000))
                    running_loss = 0.0 # Reset running loss tracker after logging
                    
    return net
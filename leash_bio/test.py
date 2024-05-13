import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd


class DaskParquetDataset(Dataset):
    def __init__(self, parquet_file, feature_columns, label_column):
        """
        Initialize the dataset using a Dask DataFrame.
        """
        self.ddf = dd.read_parquet(parquet_file, engine="pyarrow")
        self.feature_columns = feature_columns
        self.label_column = label_column

    def __len__(self):
        return len(self.ddf)

    def __getitem__(self, idx):
        """
        Fetches a single data point from the dataset.
        """
        sample = self.ddf.loc[idx].compute()
        features = pd.get_dummies(sample[self.feature_columns]).astype(float)
        x = torch.tensor(features.values, dtype=torch.float32).flatten()
        y = torch.tensor([sample[self.label_column]], dtype=torch.float32)
        return x, y


class ClassWeightedSampler(Sampler):
    """
    A sampler that accounts for class imbalance by using weights.
    """

    def __init__(self, dataset, label_column, num_samples):
        self.dataset = dataset
        self.label_column = label_column
        self.num_samples = num_samples
        self.weights = self.compute_class_weights()
        self.weight_array = self.compute_weight_array()

    def compute_weight_array(self):
        weights = np.array(
            [
                self.weights[label]
                for label in self.dataset.ddf[self.label_column].compute()
            ]
        )
        # Normalize the weights
        return weights / sum(weights)

    def compute_class_weights(self):
        label_counts = self.dataset.ddf[self.label_column].value_counts().compute()
        total_count = sum(label_counts.values)
        return {label: total_count / count for label, count in label_counts.items()}

    def __iter__(self):
        # choose indices with replacement from 2 classes
        return iter(
            np.random.choice(
                range(len(self.dataset)),
                size=self.num_samples,
                p=self.weight_array,
                replace=True,
            )
        )

    def __len__(self):
        return self.num_samples


class LogisticRegression(nn.Module):
    """
    A simple logistic regression model.
    """

    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


class Model:
    """
    A model class that encapsulates training and evaluation of the neural network.
    """

    def __init__(self, data_loader, n_features):
        self.model = LogisticRegression(n_features)
        self.criterion = nn.BCEWithLogitsLoss()  # Dynamic weights are used in sampler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.data_loader = data_loader

    def train(self, num_epochs=100):
        """
        Trains the model for a specified number of epochs.
        """
        print("Training the model...")
        for epoch in range(num_epochs):
            for inputs, targets in self.data_loader:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                # save checkpoint
                self.save(f"model_checkpoint_{epoch}.pth")

    def evaluate(self):
        """
        Evaluates the model.
        """
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        for inputs, targets in self.data_loader:
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets.squeeze())
            predicted = outputs >= 0
            accuracy = (predicted == targets).float().mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
        return total_loss / len(self.data_loader), total_accuracy / len(
            self.data_loader
        )

    def save(self, path):
        """
        Saves the model weights.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the model weights.
        """
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    parquet_file = "leash_bio/data/train.parquet"
    features_columns = [
        "buildingblock1_smiles",
        "buildingblock2_smiles",
        "buildingblock3_smiles",
        "protein_name",
    ]
    label_column = "binds"
    dataset = DaskParquetDataset(parquet_file, features_columns, label_column)
    sampler = ClassWeightedSampler(dataset, label_column, len(dataset))
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    model = Model(
        loader, len(dataset.ddf.columns) - 1
    )  # Assuming all but label_column are features
    model.train()
    print("Training complete.")
    loss, accuracy = model.evaluate()
    print(f"Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    model.save("model.pth")

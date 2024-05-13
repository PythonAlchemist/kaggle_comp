import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Sampler, Dataset
from typing import Union
from dask.dataframe import dd


class DaskParquetDataset(Dataset):
    def __init__(self, parquet_file, features_columns, label_column):
        """
        Initialize the dataset with the path to the parquet file and the columns to use.
        :param parquet_file: str, path to the parquet file.
        :param features_columns: list of str, names of the feature columns.
        :param label_column: str, name of the label column.
        """
        self.ddf = dd.read_parquet(parquet_file, engine="pyarrow")
        self.features_columns = features_columns
        self.label_column = label_column

    def __len__(self):
        """
        Returns the total number of rows in the dataframe.
        """
        return len(self.ddf)

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset.
        :param idx: int, the index of the item.
        :return: (features, label) as tensors.
        """
        # Compute only the specific row, converting Dask DataFrame to pandas DataFrame
        sample = self.ddf.loc[idx].compute()
        x = torch.tensor(sample[self.features_columns].values, dtype=torch.float32)
        y = torch.tensor(
            [sample[self.label_column]], dtype=torch.float32
        )  # Note the bracket to keep dimension
        return x, y

    @classmethod
    def compute_weights(parquet_file):
        ddf = dd.read_parquet(parquet_file, columns=["binds"])
        # Assuming 'binds' is 1 for positives and 0 for negatives and you want to upsample positives
        counts = ddf["binds"].value_counts().compute()
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]

        weights = ddf["binds"].map({0: weight_for_0, 1: weight_for_1}).compute()
        weights.to_parquet(
            "path_to_weights.parquet"
        )  # Saving the weights for later use


class WeightedSampler(Sampler):
    def __init__(self, weights_file, num_samples):
        self.weights = pd.read_parquet(weights_file)[
            "weights"
        ].values  # This loads all weights into memory; adjust if too large
        self.num_samples = num_samples

    def __iter__(self):
        return iter(
            torch.multinomial(
                torch.tensor(self.weights, dtype=torch.double),
                self.num_samples,
                replacement=True,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples


class LeashDataset:
    def __init__(self, parquet_file: str, limit: Union[int, None] = None):
        self.df = (
            pd.read_parquet(parquet_file, engine="pyarrow")
            if not limit
            else pd.read_parquet(parquet_file, engine="pyarrow")[:limit]
        )

    def __len__(self):
        return len(self.df)

    def one_hot_encode(self, column: str) -> pd.DataFrame:
        return pd.get_dummies(self.df[column])

    def transform(self) -> None:
        self.features = pd.concat(
            [
                self.one_hot_encode("buildingblock1_smiles"),
                self.one_hot_encode("buildingblock2_smiles"),
                self.one_hot_encode("buildingblock3_smiles"),
                self.one_hot_encode("protein_name"),
            ],
            axis=1,
        )
        self.features = torch.tensor(self.features.values).float()
        self.labels = torch.tensor(self.df["binds"].values).float()

        # WeightedRandomSampler to balance the classes. 0 is the majority class
        sampler = WeightedRandomSampler(
            weights=(self.labels == 0).float() + 1,
            num_samples=len(self.labels),
            replacement=True,
        )

        # count the number of 1s and 0s
        print(f"Number of 1s: {(self.labels == 1).sum()}")
        print(f"Number of 0s: {(self.labels == 0).sum()}")

        dataset = TensorDataset(self.features, self.labels)
        self.loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    def feature_col_count(self) -> int:
        return self.features.shape[1]


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # n features to 1 output

    def forward(self, x):
        return self.linear(x)


class Model:
    def __init__(self, data):
        self.n_features = data.feature_col_count()
        self.model = LogisticRegression(self.n_features)
        self.data = data
        self.set_loss_function()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def set_loss_function(self):
        # Calculate class weights
        num_positives = self.data.labels.sum()
        num_negatives = len(self.data.labels) - num_positives
        pos_weight = num_negatives / num_positives if num_positives != 0 else 1
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float)

        # Set the criterion with dynamic pos_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def train(self) -> None:
        num_epochs = 100
        for epoch in range(num_epochs):
            for inputs, targets in self.data.loader:
                # Forward pass
                outputs = self.model(inputs).squeeze()  # Remove extra dimensions
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                self.data.features
            ).squeeze()  # Consistent output handling
            loss = self.criterion(outputs, self.data.labels.float())
            predicted = outputs >= 0  # Generate binary predictions
            accuracy = (predicted == self.data.labels).float().mean()
        return loss, accuracy

    def show_predictions(self, count=5):
        print("Predictions:")
        for i in range(count):
            prediction = self.model(self.data.features[i]).item()
            true_label = self.data.labels[i].item()
            print(f"Prediction: {prediction}, True: {true_label}")


if __name__ == "__main__":
    parquet_file = "leash_bio/data/train.parquet"
    features_columns = [
        "buildingblock1_smiles",
        "buildingblock2_smiles",
        "buildingblock3_smiles",
        "protein_name",
    ]
    data = DaskParquetDataset(parquet_file, features_columns, "binds")
    loader = DataLoader(data, batch_size=32, shuffle=False)

    # model = Model(data)
    # model.train()
    # loss, accuracy = model.evaluate()
    # print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    # # show a few predictions
    # print("Predictions:")
    # model.show_predictions(10)

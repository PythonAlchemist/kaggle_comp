import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Sampler,
    Dataset,
    WeightedRandomSampler,
)
from typing import Union
from dask.dataframe import dd
from sklearn.model_selection import train_test_split
from embedding import SMILESToEmbedding
import torch.nn.functional as F


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
        self.embedder = SMILESToEmbedding()
        self.embedding_count = 0

    def __len__(self):
        return len(self.df)

    def one_hot_encode(self, column: str) -> pd.DataFrame:
        n_categories = len(self.df[column].unique())

        return F.one_hot(
            torch.tensor(
                self.df[column].astype("category").cat.codes.values, dtype=torch.int64
            ),
            n_categories,
        )

    def getEmbedding(self, smiles: str) -> torch.Tensor:
        self.embedding_count += 1
        if self.embedding_count % 100 == 0:
            print(f"Embedding count: {self.embedding_count}")
        try:
            return self.embedder(smiles)
        except Exception as e:
            print(f"Error: {e}")
            return torch.zeros(768)

    def transform(self) -> None:

        new_df = pd.DataFrame()
        new_df["protein"] = self.one_hot_encode("protein_name")

        new_df["smile_1"] = self.df["buildingblock1_smiles"].apply(self.getEmbedding)
        new_df["smile_2"] = self.df["buildingblock2_smiles"].apply(self.getEmbedding)
        new_df["smile_3"] = self.df["buildingblock3_smiles"].apply(self.getEmbedding)
        new_df["target"] = self.df["binds"]

        self.df = new_df

        # flatten the embeddings into a single tensor
        self.df["features"] = self.df.apply(
            lambda x: torch.cat([x[col] for col in new_df.columns]),
            axis=1,
        )

        # test train split
        test, train = train_test_split(self.df, test_size=0.2)

        self.train_features = torch.tensor(train["features"].values).float()
        self.train_labels = torch.tensor(train["target"].values).float()
        self.test_features = torch.tensor(test["features"].values).float()
        self.test_labels = torch.tensor(test["target"].values).float()

        # WeightedRandomSampler to balance the classes. 0 is the majority class
        sampler = WeightedRandomSampler(
            weights=(self.labels == 0).float() + 1,
            num_samples=len(self.labels),
            replacement=True,
        )

        # count the number of 1s and 0s
        print(f"Number of 1s: {(self.train_labels == 1).sum()}")
        print(f"Number of 0s: {(self.train_labels == 0).sum()}")

        dataset = TensorDataset(self.train_features, self.train_labels)
        self.loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    def feature_col_count(self) -> int:
        return self.train_features.shape[1]


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
    # features_columns = [
    #     "buildingblock1_smiles",
    #     "buildingblock2_smiles",
    #     "buildingblock3_smiles",
    #     "protein_name",
    # ]
    # data = DaskParquetDataset(parquet_file, features_columns, "binds")
    # loader = DataLoader(data, batch_size=32, shuffle=False)

    data = LeashDataset(parquet_file, limit=1_000)
    data.transform()

    test_data = LeashDataset("leash_bio/data/test.parquet")
    test_data.transform()

    model = Model(data)
    model.train()
    loss, accuracy = model.evaluate()
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    # show a few predictions
    print("Predictions:")
    model.show_predictions(10)

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from dataclasses import dataclass
from typing import Self
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    """
    Class that transforms the datasets stored as `*.csv` files into actual `Dataset`s for torch to iterate through
    """

    def __init__(self, csv_path: str, normalize: bool = False, only_impact_times: bool = False):
        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        self.only_impact_times = only_impact_times

        if self.normalize:
            cols_to_norm = [i for i in list(self.df.columns) if i not in ["X", "Y"]]
            self.df[cols_to_norm] = self.df[cols_to_norm].apply(self.__normalize)

    def __normalize(self, x):
        """
        Data normalization to be in range of 0.0 to 1.0.
        """
        y_max, y_min = 1.0, 0.0
        x_max, x_min = np.max(x), np.min(x)
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def __getitem__(self, index):
        row = self.df.iloc[index].to_numpy()

        if self.only_impact_times:
            features = row[0:6]  # extract columns [0, 5] that represent impact times
        else:
            # also use the ampliltude values as input features. This is the case of Jung et al. 2020, but not for Yu et al. 2024.
            # for my implementations this distinction is not made
            features = row[0:-2]

        label = row[-2:]
        return features, label

    def __len__(self) -> int:
        return len(self.df)


def get_dataset(dataset: Dataset, validation_split: float, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Return a shuffled Dataloader for training & validation with a given batch size from the full Dataset
    """
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create a dataset
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train, val


@dataclass
class ImpactLocation:
    x: float
    y: float
    label: str
    color: str  # for plotting


class LocationsContainer:
    """
    Class that is used as a way of organising `ImpactLocation` objects in a way where one can easily retrieve the desired one from any of its unique attributes.
    An instance of this class is created below and populated with the actual impact impact locations via the `add()` method. This ensures it acts as a singleton.
    """

    locations: list[ImpactLocation] = []

    def add(self, x: float, y: float, label: str, color: str = None) -> Self:
        if color is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            color = colors[len(self.locations)]

        new_loc = ImpactLocation(x, y, label, color)
        if new_loc not in self.locations:
            self.locations.append(new_loc)

        return self

    def from_tuple(self, loc_tuple: tuple[float, float]) -> ImpactLocation:
        for loc in self.locations:
            if (loc.x, loc.y) == loc_tuple:
                return loc
        raise IndexError

    def from_xy(self, x: float, y: float) -> ImpactLocation:
        return self.from_tuple((x, y))

    def from_label(self, label: str) -> ImpactLocation:
        for loc in self.locations:
            if loc.label == label:
                return loc
        raise IndexError

    def from_index(self, index: int) -> ImpactLocation:
        return self.locations[index]

    def get_labels(self) -> list[str]:
        labels = {*[loc.label for loc in self.locations]}
        return list(labels)


# Acts as singleton when importing module
locations = LocationsContainer()
locations.add(0, 0, "center")
locations.add(0, 60, "front")
locations.add(0, -60, "back")

from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class AerialData(Dataset):
    """
    Custom PyTorch Dataset for loading aerial image classification data 
    from a CSV file.

    The CSV file must have two columns:
        - data_path: full path to the image file
        - label: class label of the image

    Args:
        path_csv (str): Path to the CSV file containing image paths and labels.
        transform (torchvision.transforms.Compose, optional): Transformations 
            to apply to the images (e.g., resizing, normalization). Defaults to None.

    Attributes:
        image_sample (pd.DataFrame): DataFrame storing image paths and labels.
        tsform (torchvision.transforms.Compose | None): Transform pipeline.
        classes (list[str]): Sorted list of unique class labels.
        class_to_idx (dict): Mapping from class name to integer index.
    """
    def __init__(self, path_csv: str, transform: T.Compose | None = None) -> None:
        self.image_sample = pd.read_csv(path_csv)
        self.tsform = transform

        self.classes = sorted(self.image_sample["label"].unique())
        self.class_to_idx = {s: i for i, s in enumerate(self.classes)}

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset

        Returns:
            int: Number of samples
        """
        return len(self.image_sample)

    def __getitem__(self, idx):
        """
        Load and return a single sample (image and label) from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, int]: 
                - Image tensor after transformations.
                - Integer label corresponding to the class.
        """
        img = Image.open(self.image_sample.iloc[idx, 0]).convert("RGB")

        label = self.image_sample.iloc[idx, 1]
        label = self.class_to_idx[label]

        if self.tsform:
            img = self.tsform(img)

        return img, label

def get_dataloader(
        data: Dataset, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        num_workers: int = 4, 
        **kwargs
) -> tuple[DataLoader, list[str]]:
    """
    Create a PyTorch DataLoader for a dataset.

    Args:
        data (torch.utils.data.Dataset): Dataset to wrap with the DataLoader.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
        **kwargs: Additional arguments passed to `torch.utils.data.DataLoader`.

    Returns:
        tuple:
            - DataLoader: DataLoader object for batching and shuffling the dataset.
            - list[str]: List of class names from the dataset.
    """
    return DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        **kwargs
    ), data.classes

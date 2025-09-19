import cv2
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader

class AerialData(Dataset):
    def __init__(self, path_csv: str, transform: A.Compose | None = None) -> None:
        self.image_sample = pd.read_csv(path_csv)
        self.transform = transform

        self.classes = sorted(self.image_sample["label"].unique())
        self.class_to_idx = {s: i for i, s in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.image_sample)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_sample.iloc[idx, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.image_sample.iloc[idx, 1]
        label = self.class_to_idx[label]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label

def get_dataloader(data: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, **kwargs) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

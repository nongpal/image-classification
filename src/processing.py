from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class AerialData(Dataset):
    def __init__(self, path_csv: str, transform: T.Compose | None = None) -> None:
        self.image_sample = pd.read_csv(path_csv)
        self.tsform = transform

        self.classes = sorted(self.image_sample["label"].unique())
        self.class_to_idx = {s: i for i, s in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.image_sample)

    def __getitem__(self, idx):
        img = Image.open(self.image_sample.iloc[idx, 0]).convert("RGB")

        label = self.image_sample.iloc[idx, 1]
        label = self.class_to_idx[label]

        if self.tsform:
            img = self.tsform(img)
            print("After transform:", type(img))

        return img, label

def get_dataloader(data: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, **kwargs) -> tuple[DataLoader, list[str]]:
    return DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        **kwargs
    ), data.classes

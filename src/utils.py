import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_labels(path: str) -> list[str]:
    return [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

def get_data_path(path: str, labels: list[str]) -> tuple[list[str], list[str]]:
    all_labels = list()
    all_paths = list()

    for label in labels:
        folder = os.path.join(path, label)
        for file in os.listdir(folder):
            if file.endswith((".png", ".jpg", ".jpeg")):
                all_paths.append(os.path.join(folder, file))
                all_labels.append(label)
    return all_paths, all_labels

def make_file(path: str, is_split: bool = False, output_dir: str = "."):
    labels = get_labels(path)
    data_paths, data_labels = get_data_path(path, labels)

    df = pd.DataFrame({"data_path": data_paths, "label": data_labels})

    if not is_split:
        df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
        return df

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train_df, val_df, test_df

if __name__ == "__main__":
    path = "data/Aerial_Landscapes"
    train_df, val_df, test_df = make_file(path, is_split=True, output_dir="data/Aerial_Landscapes")
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

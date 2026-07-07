

import os
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, split_name, features_root="./extracted_features_5fold", fold=None):
        """Implementation details are provided by the code below."""
        if fold is not None and split_name in ("train", "val"):

            self.features_dir = os.path.join(features_root, f"fold{fold}", split_name)
        else:

            self.features_dir = os.path.join(features_root, split_name)

        if not os.path.isdir(self.features_dir):
            raise FileNotFoundError(f"Feature directory does not exist: {self.features_dir}")


        self.sample_ids = [f[:-3] for f in os.listdir(self.features_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        feature_path = os.path.join(self.features_dir, f"{sample_id}.pt")
        try:

            data = torch.load(feature_path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(feature_path, map_location="cpu")
        return {
            "acoustic_layers": data["acoustic_layers"],     # [24,1024]
            "linguistic_feat": data["linguistic_feat"],     # [1024]
            "label": data["label"]
        }


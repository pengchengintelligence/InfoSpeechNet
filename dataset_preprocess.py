






import os
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, split_name, features_root="./extracted_features_5fold", fold=None):
        """
        split_name: "train" / "val"
        features_root: 预处理特征根目录

        现在的五折预处理脚本会导出：
            extracted_features_5fold_merged/fold1/train/*.pt
            extracted_features_5fold_merged/fold1/val/*.pt
            ...
        所以这里也做了对应。

        仍然保留一个向下兼容的分支：如果没有传 fold，就去
            features_root/{split_name}
        下面找，这样你旧的单份特征也还能用。

        每个样本的 .pt 文件包含：
          - "acoustic_layers": [24,1024]
          - "linguistic_feat": [1024]
          - "label": 标量
        """
        if fold is not None and split_name in ("train", "val"):
            # 新的五折目录：./extracted_features_5fold_merged/fold{fold}/train(or val)
            self.features_dir = os.path.join(features_root, f"fold{fold}", split_name)
        else:
            # 向下兼容：./extracted_features_5fold_merged/train
            self.features_dir = os.path.join(features_root, split_name)

        if not os.path.isdir(self.features_dir):
            raise FileNotFoundError(f"特征目录不存在: {self.features_dir}")

        # 记录所有样本 id（不带 .pt）
        self.sample_ids = [f[:-3] for f in os.listdir(self.features_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        feature_path = os.path.join(self.features_dir, f"{sample_id}.pt")
        try:
            # 新版 torch.load 可能要加 weights_only
            data = torch.load(feature_path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(feature_path, map_location="cpu")
        return {
            "acoustic_layers": data["acoustic_layers"],     # [24,1024]
            "linguistic_feat": data["linguistic_feat"],     # [1024]
            "label": data["label"]                          # 标量 (int/long)
        }


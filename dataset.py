import os
import pandas as pd
import torch
import librosa
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class ADDataset(Dataset):
    def __init__(
        self,
        label_path=None,
        data_type=None,
        bert_model="../../bert-large-uncased",
        max_audio_length=2483963,
        indices=None,
        df=None,
    ):
        """
        支持两种用法：

        1) 老用法（保持兼容）：
            label_path=".../train_labels.csv", data_type="train"
           会从固定的 train / test 目录里去找文件。

        2) 新用法（给五折合并后的 DataFrame）：
            df=all_df   # 必须包含: id, label, split(train/test)
            indices=... # 某一折的 train_idx / val_idx
           这样每一行都会按行里的 split 去对应目录取文件。

        参数说明
        --------
        df: pandas.DataFrame, 必须含列 ["id", "label", "split"]，其中 split ∈ {"train", "test"}
        indices: 若给定，只取 df/CSV 中指定行号（用于K折）
        """
        self.max_audio_length = max_audio_length

        # ========= 新用法：直接给 df =========
        if df is not None:
            # df 里要有 id / label / split
            if indices is not None:
                self.labels = df.iloc[indices].reset_index(drop=True)
            else:
                self.labels = df.reset_index(drop=True)
            self.use_merged_df = True
        else:
            # ========= 老用法：路径 + data_type =========
            if label_path is None or data_type is None:
                raise ValueError("如果没有传 df，就必须同时传 label_path 和 data_type 才能定位数据。")
            all_labels = pd.read_csv(label_path)
            if indices is not None:
                self.labels = all_labels.iloc[indices].reset_index(drop=True)
            else:
                self.labels = all_labels
            self.use_merged_df = False
            self.data_type = data_type  # "train" 或 "test"

        # 检查 BERT 模型路径
        if not os.path.exists(bert_model):
            raise FileNotFoundError(f"BERT模型目录不存在: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, local_files_only=True)

    def __len__(self):
        return len(self.labels)

    def _build_paths(self, split: str):
        """
        给定一行里的 split(train/test)，构建真实的音频/文本目录。
        """
        audio_dir = os.path.join("../../data1", split, f"{split}_16k")
        transcript_dir = os.path.join("../../data1", split, f"{split}_transcriptions")
        if not os.path.isdir(audio_dir):
            raise NotADirectoryError(
                f"音频目录不存在: {audio_dir}\n"
                f"请确认路径是否为: data1/{split}/{split}_16k"
            )
        if not os.path.isdir(transcript_dir):
            raise NotADirectoryError(
                f"文本目录不存在: {transcript_dir}\n"
                f"请确认路径是否为: data1/{split}/{split}_transcriptions"
            )
        return audio_dir, transcript_dir

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        sample_id = item["id"]
        label = torch.tensor(item["label"], dtype=torch.float32)

        # 判定这一行到底来自 train 还是 test
        if self.use_merged_df:
            split = item["split"]  # 行里自带
        else:
            split = self.data_type  # 老用法：构造时就指定了

        audio_dir, transcript_dir = self._build_paths(split)

        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
        transcript_path = os.path.join(transcript_dir, f"{sample_id}.txt")

        # 加载音频
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except FileNotFoundError:
            raise FileNotFoundError(f"音频文件缺失: {audio_path}")
        audio = torch.tensor(audio, dtype=torch.float32)

        # 统一音频长度
        if len(audio) > self.max_audio_length:
            audio = audio[: self.max_audio_length]
        else:
            audio = torch.nn.functional.pad(
                audio,
                (0, self.max_audio_length - len(audio)),
                mode="constant",
            )

        # 加载文本
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"文本文件缺失: {transcript_path}")

        text_token = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "id": sample_id,
            "audio": audio,
            "text_input_ids": text_token["input_ids"].squeeze(0),
            "text_attention_mask": text_token["attention_mask"].squeeze(0),
            "label": label,
            "split": split,
        }

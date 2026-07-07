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
        """Implementation details are provided by the code below."""
        self.max_audio_length = max_audio_length


        if df is not None:

            if indices is not None:
                self.labels = df.iloc[indices].reset_index(drop=True)
            else:
                self.labels = df.reset_index(drop=True)
            self.use_merged_df = True
        else:

            if label_path is None or data_type is None:
                raise ValueError("label_path and data_type must both be provided when df is not provided.")
            all_labels = pd.read_csv(label_path)
            if indices is not None:
                self.labels = all_labels.iloc[indices].reset_index(drop=True)
            else:
                self.labels = all_labels
            self.use_merged_df = False
            self.data_type = data_type


        if not os.path.exists(bert_model):
            raise FileNotFoundError(f"BERT model directory does not exist: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, local_files_only=True)

    def __len__(self):
        return len(self.labels)

    def _build_paths(self, split: str):
        """Implementation details are provided by the code below."""
        audio_dir = os.path.join("../../data1", split, f"{split}_16k")
        transcript_dir = os.path.join("../../data1", split, f"{split}_transcriptions")
        if not os.path.isdir(audio_dir):
            raise NotADirectoryError(
                f"Audio directory does not exist: {audio_dir}\n"
                f"Expected path: data1/{split}/{split}_16k"
            )
        if not os.path.isdir(transcript_dir):
            raise NotADirectoryError(
                f"Transcript directory does not exist: {transcript_dir}\n"
                f"Expected path: data1/{split}/{split}_transcriptions"
            )
        return audio_dir, transcript_dir

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        sample_id = item["id"]
        label = torch.tensor(item["label"], dtype=torch.float32)


        if self.use_merged_df:
            split = item["split"]
        else:
            split = self.data_type

        audio_dir, transcript_dir = self._build_paths(split)

        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
        transcript_path = os.path.join(transcript_dir, f"{sample_id}.txt")


        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file is missing: {audio_path}")
        audio = torch.tensor(audio, dtype=torch.float32)


        if len(audio) > self.max_audio_length:
            audio = audio[: self.max_audio_length]
        else:
            audio = torch.nn.functional.pad(
                audio,
                (0, self.max_audio_length - len(audio)),
                mode="constant",
            )


        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file is missing: {transcript_path}")

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

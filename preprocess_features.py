import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset import ADDataset
from feature_extractor import AudioTextFeatureExtractor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../../data1"
TRAIN_LABEL_PATH = os.path.join(DATA_ROOT, "train", "adresso21-train-labels.csv")
TEST_LABEL_PATH  = os.path.join(DATA_ROOT, "test",  "adresso21_test_labels.csv")

BERT_DIR    = "../../bert-large-uncased"
W2V_DIR     = "../../wavlm-large"

OUTPUT_DIR  = "extracted_features_5fold_seed43-1"

BATCH_SIZE  = 1
NUM_WORKERS = 4
PIN_MEMORY  = True

N_FOLDS = 5
SEED    = 43


def extract_and_save_features(dataloader, model, save_dir):
    """Implementation details are provided by the code below."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc=f"Processing {os.path.basename(save_dir)}"):
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        text_input_ids = batch["text_input_ids"].to(DEVICE, non_blocking=True)
        text_attention_mask = batch["text_attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"]

        acoustic_layers, linguistic_feat = model(audio, text_input_ids, text_attention_mask)
        B = acoustic_layers.size(0)

        for i in range(B):
            sample_id = batch["id"][i]
            save_path = os.path.join(save_dir, f"{sample_id}.pt")
            torch.save(
                {
                    "acoustic_layers": acoustic_layers[i].detach().cpu(),
                    "linguistic_feat": linguistic_feat[i].detach().cpu(),
                    "label":           labels[i].detach().cpu() if torch.is_tensor(labels[i]) else labels[i],
                },
                save_path,
            )


def build_dataloader_from_df(df: pd.DataFrame, indices, bert_dir: str) -> DataLoader:
    """Implementation details are provided by the code below."""
    dataset = ADDataset(
        df=df,
        bert_model=bert_dir,
        indices=indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return loader


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    train_df = pd.read_csv(TRAIN_LABEL_PATH)
    train_df["split"] = "train"

    test_df = pd.read_csv(TEST_LABEL_PATH)
    test_df["split"] = "test"


    all_df = pd.concat([train_df, test_df], ignore_index=True)
    y = all_df["label"].values


    model = AudioTextFeatureExtractor(
        wav2vec_dir=W2V_DIR,
        bert_dir=BERT_DIR,
        num_layers=24,
        feat_dim=1024,
        freeze_backbones=True,
    ).to(DEVICE)
    model.eval()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_df, y), start=1):
        print(f"\n===== Processing fold {fold_idx}/{N_FOLDS} =====")

        fold_train_dir = os.path.join(OUTPUT_DIR, f"fold{fold_idx}", "train")
        fold_val_dir   = os.path.join(OUTPUT_DIR, f"fold{fold_idx}", "val")
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_val_dir, exist_ok=True)


        train_loader_fold = build_dataloader_from_df(all_df, train_idx, BERT_DIR)
        val_loader_fold   = build_dataloader_from_df(all_df, val_idx,   BERT_DIR)

        print(f"[Fold {fold_idx}] Extracting train features, total {len(train_idx)} samples...")
        extract_and_save_features(train_loader_fold, model, fold_train_dir)

        print(f"[Fold {fold_idx}] Extracting validation features, total {len(val_idx)} samples...")
        extract_and_save_features(val_loader_fold, model, fold_val_dir)

        print(f"[Fold {fold_idx}] Finished and saved to {os.path.join(OUTPUT_DIR, f'fold{fold_idx}')}")

    print(f"\nAll features have been saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

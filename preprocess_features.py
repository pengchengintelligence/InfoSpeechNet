import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset import ADDataset
from feature_extractor import AudioTextFeatureExtractor

# ========================== 基本配置 ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../../data1"
TRAIN_LABEL_PATH = os.path.join(DATA_ROOT, "train", "adresso21-train-labels.csv")
TEST_LABEL_PATH  = os.path.join(DATA_ROOT, "test",  "adresso21_test_labels.csv")

BERT_DIR    = "../../bert-large-uncased"
W2V_DIR     = "../../wavlm-large"

OUTPUT_DIR  = "extracted_features_5fold_seed43-1"   # 新的输出根目录

BATCH_SIZE  = 1
NUM_WORKERS = 4
PIN_MEMORY  = True

N_FOLDS = 5
SEED    = 43


def extract_and_save_features(dataloader, model, save_dir):
    """
    导出每个样本：
      - 'acoustic_layers' : [24,1024]
      - 'linguistic_feat' : [1024]
      - 'label'
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc=f"处理 {os.path.basename(save_dir)}"):
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
    """
    用合并后的 df + 指定行号来构建 DataLoader
    df 必须有 id, label, split
    """
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

    # 1) 读训练 + 测试标签，并标记它来自 train 还是 test
    train_df = pd.read_csv(TRAIN_LABEL_PATH)
    train_df["split"] = "train"

    test_df = pd.read_csv(TEST_LABEL_PATH)
    test_df["split"] = "test"

    # 合并成一个整体，再做一次分层 5 折
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    y = all_df["label"].values

    # 2) 构建一次特征模型
    model = AudioTextFeatureExtractor(
        wav2vec_dir=W2V_DIR,
        bert_dir=BERT_DIR,
        num_layers=24,
        feat_dim=1024,
        freeze_backbones=True,
    ).to(DEVICE)
    model.eval()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # 3) 对每一折都导一次 train+val
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_df, y), start=1):
        print(f"\n===== 开始处理第 {fold_idx}/{N_FOLDS} 折 =====")

        fold_train_dir = os.path.join(OUTPUT_DIR, f"fold{fold_idx}", "train")
        fold_val_dir   = os.path.join(OUTPUT_DIR, f"fold{fold_idx}", "val")
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_val_dir, exist_ok=True)

        # 这一折的 dataloader
        train_loader_fold = build_dataloader_from_df(all_df, train_idx, BERT_DIR)
        val_loader_fold   = build_dataloader_from_df(all_df, val_idx,   BERT_DIR)

        print(f"[Fold {fold_idx}] 提取训练(train)特征，共 {len(train_idx)} 条...")
        extract_and_save_features(train_loader_fold, model, fold_train_dir)

        print(f"[Fold {fold_idx}] 提取验证(val)特征，共 {len(val_idx)} 条...")
        extract_and_save_features(val_loader_fold, model, fold_val_dir)

        print(f"[Fold {fold_idx}] 已完成并保存至 {os.path.join(OUTPUT_DIR, f'fold{fold_idx}')}")

    print(f"\n所有特征已成功保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

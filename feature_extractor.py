import os
import torch
import torch.nn as nn
from transformers import (
    WavLMModel,     # 用 WavLM 模型
    BertModel,
    BertTokenizer,
)

class AudioTextFeatureExtractor(nn.Module):
    """
    用于离线特征导出（不包含 TSAC）：
      - 声学侧：Wav2Vec2 -> 最近 24 层隐藏态 -> 时间维均值 -> [B, 24, 1024]
      - 文本侧：BERT last_hidden_state -> token 维均值 -> [B, 1024]  (bert-large)
      - forward 返回：(acoustic_layers, linguistic_feat)
    """

    def __init__(
        self,
        wav2vec_dir: str = "../../wavlm-large",
        bert_dir: str = "../../bert-large-uncased",
        num_layers: int = 24,
        feat_dim: int = 1024,
        freeze_backbones: bool = True,
    ):
        super().__init__()

        # 路径检查
        if not os.path.exists(wav2vec_dir):
            raise FileNotFoundError(f"[AudioTextFeatureExtractor] Wav2Vec2 模型目录不存在: {wav2vec_dir}")
        if not os.path.exists(bert_dir):
            raise FileNotFoundError(f"[AudioTextFeatureExtractor] BERT 模型目录不存在: {bert_dir}")

        # 预训练模型与处理器（本地加载）
        # 预训练模型（本地加载 WavLM）
        self.wav2vec = WavLMModel.from_pretrained(
            wav2vec_dir, local_files_only=True
        )

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            bert_dir, local_files_only=True
        )
        self.bert = BertModel.from_pretrained(
            bert_dir, local_files_only=True
        )

        # 常量（控制导出维度）
        self.num_layers = num_layers   # 24
        self.feat_dim   = feat_dim     # 1024

        # 预处理阶段通常冻结骨干以节省显存、确保稳定
        if freeze_backbones:
            for p in self.wav2vec.parameters():
                p.requires_grad = False
            for p in self.bert.parameters():
                p.requires_grad = False

        self.eval()

    # def get_wav_processor(self):
    #     return self.wav2vec_processor

    def get_bert_tokenizer(self):
        return self.bert_tokenizer

    @torch.no_grad()
    def forward(self, audio, text_input_ids, text_attention_mask):
        """
        返回：
          acoustic_layers: FloatTensor [B, 24, 1024]  (W2V 最近 24 层的时间均值)
          linguistic_feat: FloatTensor [B, 1024]      (BERT token 均值；bert-large)
        """
        # 声学侧：取最近 24 层隐藏态，沿时间维均值
        wav_outputs = self.wav2vec(audio, output_hidden_states=True)
        # hidden_states: tuple(len = encoder_layers + 1)，每项 [B, T, 1024]
        stacked = torch.stack(
            wav_outputs.hidden_states[-self.num_layers:], dim=1
        )  # [B, 24, T, 1024]
        acoustic_layers = stacked.mean(dim=2)            # [B, 24, 1024]

        # 文本侧：last_hidden_state 沿 token 维均值
        bert_outputs = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        linguistic_feat = bert_outputs.last_hidden_state.mean(dim=1)  # [B, 1024]

        return acoustic_layers, linguistic_feat

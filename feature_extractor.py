import os
import torch
import torch.nn as nn
from transformers import (
    WavLMModel,
    BertModel,
    BertTokenizer,
)

class AudioTextFeatureExtractor(nn.Module):
    def __init__(
        self,
        wav2vec_dir: str = "../../wavlm-large",
        bert_dir: str = "../../bert-large-uncased",
        num_layers: int = 24,
        feat_dim: int = 1024,
        freeze_backbones: bool = True,
    ):
        super().__init__()


        if not os.path.exists(wav2vec_dir):
            raise FileNotFoundError(f"[AudioTextFeatureExtractor] Wav2Vec2 model directory does not exist: {wav2vec_dir}")
        if not os.path.exists(bert_dir):
            raise FileNotFoundError(f"[AudioTextFeatureExtractor] BERT model directory does not exist: {bert_dir}")


        self.wav2vec = WavLMModel.from_pretrained(
            wav2vec_dir, local_files_only=True
        )

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            bert_dir, local_files_only=True
        )
        self.bert = BertModel.from_pretrained(
            bert_dir, local_files_only=True
        )


        self.num_layers = num_layers   # 24
        self.feat_dim   = feat_dim     # 1024


        if freeze_backbones:
            for p in self.wav2vec.parameters():
                p.requires_grad = False
            for p in self.bert.parameters():
                p.requires_grad = False

        self.eval()
        
    def get_bert_tokenizer(self):
        return self.bert_tokenizer

    @torch.no_grad()
    def forward(self, audio, text_input_ids, text_attention_mask):
        """Implementation details are provided by the code below."""

        wav_outputs = self.wav2vec(audio, output_hidden_states=True)

        stacked = torch.stack(
            wav_outputs.hidden_states[-self.num_layers:], dim=1
        )  # [B, 24, T, 1024]
        acoustic_layers = stacked.mean(dim=2)            # [B, 24, 1024]


        bert_outputs = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        linguistic_feat = bert_outputs.last_hidden_state.mean(dim=1)  # [B, 1024]

        return acoustic_layers, linguistic_feat

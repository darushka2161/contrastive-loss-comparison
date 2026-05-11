import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Literal


class SentenceEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean"):
        super().__init__()
        assert pooling in ("cls", "mean"), f"pooling must be 'cls' or 'mean', got {pooling}"
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        last_hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            embeddings = last_hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return embeddings

    def encode_batch(self, encoding: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", None)
        return self.forward(input_ids, attention_mask, token_type_ids)

    @torch.no_grad()
    def encode_sentences(self, sentences, tokenizer, batch_size: int = 64,
                         max_length: int = 128, device: str = "cpu") -> torch.Tensor:
        self.eval()
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = self.encode_batch(enc)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

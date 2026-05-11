import torch
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict


def get_embeddings_for_visualization(model, tokenizer, sentences: List[str],
                                     device, batch_size: int = 32,
                                     max_length: int = 128) -> np.ndarray:
    embeddings = model.encode_sentences(sentences, tokenizer, batch_size, max_length, str(device))
    return F.normalize(embeddings, dim=-1).numpy()


def reduce_with_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def reduce_with_tsne(embeddings: np.ndarray, n_components: int = 2,
                     perplexity: float = 30.0) -> np.ndarray:
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity,
                n_iter=1000, verbose=0)
    return tsne.fit_transform(embeddings)

from abc import ABC, abstractmethod
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class CLIPEmbedder(Embedder):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        from transformers import CLIPProcessor, CLIPModel

        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            self.device = "cpu"
        else:
            self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features.cpu().numpy().flatten()

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device if device is not None else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

class GTEEmbedder(Embedder):
    def __init__(self, model_name="Alibaba-NLP/gte-large-en-v1.5", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().flatten()

# You can add more embedder classes here, e.g., for Word2Vec, FastText, etc.

embedders = {
    "clip": CLIPEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder,
    "gte": GTEEmbedder,
    # Add more embedders here as they are implemented
}

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedding_Manager:
    def __init__( self, model_name="all-MiniLM-L6-v2" ):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading {self.model_name}")
            self.model = SentenceTransformer(self.model_name )

        except Exception as e:
            raise Exception(f"Embedding loading failed: {e}")

    def generate_embedding(self,texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        embeddings = self.model.encode(texts, batch_size=32,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
        return embeddings
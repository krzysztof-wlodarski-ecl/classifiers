"""
Util to embed and cache whole directories to enable faster start.
"""
from utils.embedding import EmbeddingEngine

root = "../images/"

engine = EmbeddingEngine()
engine.embed_path(root)

# cache is already saved at this point, enjoy
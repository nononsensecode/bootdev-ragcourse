from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import load_movies, Movie


class MultimodalSearch:

    def __init__(self, documents: list[Movie] = [], model_name="clip-ViT-B-32") -> None:
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.model = SentenceTransformer(f'sentence-transformers/{model_name}')
        if documents:
            self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.embeddings = []

    def embed_image(self, image_path: str):
        img = Image.open(image_path)
        embedding = self.model.encode([img])
        return embedding[0]
    
    def search_with_image(self, image_path: str):
        embedding = self.embed_image(image_path)
        results = []
        for (doc_embedding, doc) in zip(self.embeddings, self.documents):
            sim_score = cosine_similarity(embedding, doc_embedding)
            results.append({
                "id": doc['id'],
                "title": doc['title'],
                "description": doc['description'],
                "score": sim_score
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]


def verify_image_embedding(image_path: str):
    print("verifying image embedding...")
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    movies = load_movies()
    multimodal_search = MultimodalSearch(documents=movies)
    return multimodal_search.search_with_image(image_path)


def cosine_similarity(v1, v2):
    dotproduct = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dotproduct / (norm_v1 * norm_v2)

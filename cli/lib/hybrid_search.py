import os

from .keyword_search import InvertedIndex
from .search_utils import Movie, SearchResult, load_movies
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_normalized_scores = normalize_scores(
            [result["score"] for result in bm25_results]
        )
        semantic_normalized_scores = normalize_scores(
            [result["score"] for result in semantic_results]
        )

        combined_scores: dict[int, dict] = {}

        for score, result in zip(bm25_normalized_scores, bm25_results):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if combined_scores[doc_id]["bm25_score"] < score:
                combined_scores[doc_id]["bm25_score"] = score

        for score, result in zip(semantic_normalized_scores, semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if combined_scores[doc_id]["semantic_score"] < score:
                combined_scores[doc_id]["semantic_score"] = score

        hybrid_results: list[dict] = []
        for doc_id, result in combined_scores.items():
            hybrid_results.append(
                {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": result["bm25_score"],
                    "semantic_score": result["semantic_score"],
                    "hybrid_score": hybrid_score(
                        result["bm25_score"], result["semantic_score"], alpha
                    ),
                }
            )
        return sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True)[
            :limit
        ]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined_rank_results: dict[int, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_rank_results:
                combined_rank_results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            if combined_rank_results[doc_id]["bm25_rank"] is None:
                combined_rank_results[doc_id]["bm25_rank"] = rank

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_rank_results:
                combined_rank_results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            if combined_rank_results[doc_id]["semantic_rank"] is None:
                combined_rank_results[doc_id]["semantic_rank"] = rank

        hybrid_results: list[dict] = []
        for doc_id, result in combined_rank_results.items():
            total_rrf = 0.0

            if result["bm25_rank"] is not None:
                total_rrf += rrf_score(result["bm25_rank"], k)
            if result["semantic_rank"] is not None:
                total_rrf += rrf_score(result["semantic_rank"], k)

            hybrid_results.append(
                {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": result["bm25_rank"],
                    "semantic_rank": result["semantic_rank"],
                    "rrf_score": total_rrf,
                }
            )

        return sorted(hybrid_results, key=lambda x: x["rrf_score"], reverse=True)[
            :limit
        ]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    diff = max_score - min_score
    if diff == 0:
        return [1.0] * len(scores)

    return [(score - min_score) / diff for score in scores]


def weighted_search(query: str, alpha: float, limit: int) -> list[dict]:
    movies = load_movies()
    search = HybridSearch(movies)
    return search.weighted_search(query, alpha, limit)


def rrf_search(query: str, k: int, limit: int) -> list[dict]:
    movies = load_movies()
    search = HybridSearch(movies)
    return search.rrf_search(query, k, limit)

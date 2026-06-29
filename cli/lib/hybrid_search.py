import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from sentence_transformers import CrossEncoder

from .keyword_search import InvertedIndex
from .search_utils import Movie, SearchResult, load_movies
from .semantic_search import ChunkedSemanticSearch

load_dotenv()


def generate_response(
    contents: str,
    provider: Optional[str] = "openai",
    model: Optional[str] = "gpt-4-turbo",
) -> str:
    response = ""
    match provider:
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set")
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model or "gpt-4-turbo", input=contents
            )
            response = response.output_text
            print(f"Openai response: {response}")
        case "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable is not set")
            client = genai.Client()
            response = client.models.generate_content(
                model=model or "gemma-4-31b-it", contents=contents
            )
            response = response.text or ""
        case _:
            raise RuntimeError("there is no such provider")

    return response


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
        print(
            f"bm25-results: {', '.join([result["title"] for result in bm25_results[:10]])}"
        )

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        print(
            f"semantic-results: {', '.join([result["title"] for result in semantic_results[:10]])}"
        )

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
        print(
            f"bm25-results: {', '.join([result["title"] for result in bm25_results[:10]])}"
        )

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        print(
            f"semantic-results: {', '.join([result["title"] for result in semantic_results[:10]])}"
        )

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


def enhance_query(
    query: str,
    provider: Optional[str] = "openai",
    model: Optional[str] = "gpt-4-turbo",
    enhance_method: Optional[str] = None,
) -> str:
    prompt: Optional[str] = None
    match enhance_method:
        case None:
            return query
        case "spell":
            prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        case "rewrite":
            prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        case "expand":
            prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
        case _:
            return query

    return generate_response(provider=provider, model=model, contents=prompt)


def rerank(
    search: HybridSearch,
    query: str,
    k: int,
    limit: int,
    rerank_method: Optional[str] = None,
    provider: Optional[str] = "openai",
    model: Optional[str] = "gpt-4-turbo",
) -> list[dict]:
    print(f"Rerank method: {rerank_method}")
    match rerank_method:
        case None:
            return search.rrf_search(query, k, limit)
        case "individual":
            new_limit = 5 * limit
            results = search.rrf_search(query, k, new_limit)
            for index, doc in enumerate(results):
                prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""
                response = generate_response(
                    provider="openai", model="gpt-4-turbo", contents=prompt
                )
                results[index]["rerank_score"] = float(response or "0.000")
                time.sleep(3)
        case "batch":
            new_limit = 5 * limit
            results = search.rrf_search(query, k, new_limit)
            doc_list_str = json.dumps(
                [
                    {
                        "id": result["id"],
                        "title": result["title"],
                        "description": result["document"],
                    }
                    for result in results
                ]
            )
            prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
            response = generate_response(
                provider=provider, model=model, contents=prompt
            )
            ids = json.loads(response or "[]")
            if ids:
                if len(results) != len(ids):
                    return results[:limit]
                position = {movie_id: i for i, movie_id in enumerate(ids)}
                sorted_movies = sorted(results, key=lambda m: position[m["id"]])
                return sorted_movies[:limit]
            else:
                return results[:limit]
        case "cross_encoder":
            new_limit = limit * 5
            results = search.rrf_search(query, k, new_limit)
            print(f"Cross encoder input results: {", ".join([result["title"] for result in results])}")
            pairs = [
                [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
                for doc in results
            ]
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            for doc, score in zip(results, scores):
                doc["cross_encoder_score"] = score
            sorted_by_cross_encoder_score = sorted(
                results, key=lambda doc: doc["cross_encoder_score"], reverse=True
            )
            print(
                f"Cross encoder output results: {", ".join([result["title"] for result in sorted_by_cross_encoder_score[:limit]])}"
            )
            return sorted_by_cross_encoder_score[:limit]
        case _:
            return search.rrf_search(query, k, limit)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:limit]


def rrf_search(
    query: str,
    k: int,
    limit: int,
    enhance_method: Optional[str] = None,
    rerank_method: Optional[str] = None,
) -> dict:
    movies = load_movies()
    search = HybridSearch(movies)
    original_query = query
    enhanced_query = enhance_query(original_query, enhance_method)
    reranked_results = rerank(search, query, k, limit, rerank_method)
    print(f"Original query: {original_query}")
    print(f"enhance method: {enhance_method or "N/A"}")
    print(f"re-rank method: {rerank_method or "N/A"}")
    print(f"Enhanced query: {enhanced_query}")
    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance_method,
        "query": query,
        "k": k,
        "results": reranked_results,
    }

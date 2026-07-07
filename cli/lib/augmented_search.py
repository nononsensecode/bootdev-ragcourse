from lib.hybrid_search import HybridSearch, generate_response
from lib.search_utils import load_movies


def augmented_search(query: str, k: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(documents=movies)
    search_results = hybrid_search.rrf_search(query=query, k=k)
    prompt = prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{search_results[:k]}

Answer:"""
    llm_response = generate_response(contents=prompt)
    response = {}
    response["titles"] = [result["title"] for result in search_results[:k]]
    response["llm_response"] = llm_response
    return response


def summarize_search(query: str, k: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query=query, k=k)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{search_results[:k]}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    llm_response = generate_response(contents=prompt)
    response = {}
    response["titles"] = [result["title"] for result in search_results[:k]]
    response["llm_response"] = llm_response
    return response

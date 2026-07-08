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

def citations_search(query: str, limit: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(documents=movies)
    search_results = hybrid_search.rrf_search(query=query, k=60, limit=limit)
    prompt = (
        prompt
    ) = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{search_results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    llm_response = generate_response(contents=prompt)
    response = {}
    response["titles"] = [result["title"] for result in search_results]
    response["llm_response"] = llm_response
    return response

def question_answer(question: str, limit: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(documents=movies)
    search_results = hybrid_search.rrf_search(query=question, k=60, limit=limit)
    prompt = (
        prompt
    ) = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{search_results}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    llm_response = generate_response(contents=prompt)
    response = {}
    response["titles"] = [result["title"] for result in search_results]
    response["llm_response"] = llm_response
    return response
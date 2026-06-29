from lib.hybrid_search import HybridSearch
from lib.search_utils import load_golden_dataset, load_movies


def evaluate_golden_dataset(limit: int) -> list[dict]:
    dataset = load_golden_dataset()
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    evaluation_results: list[dict] = []
    for data in dataset["test_cases"]:
        query = data["query"]
        search_response = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [doc["title"] for doc in search_response]
        relevant_docs = data["relevant_docs"]

        top_k = retrieved_titles[:limit]
        relevant_count = sum(
            [1 for title in relevant_docs if title in top_k]
        )
        precision = relevant_count / limit if limit else 0.0
        recall = relevant_count / len(relevant_docs)
        precision_recall = precision + recall
        f1_score = 2 * (precision * recall) / precision_recall if precision_recall else 0.0
        # if query == "children's animated bear adventure":
        #     print()
        #     print()
        #     print(f"Retrieved titles: {', '.join(retrieved_titles)}")
        #     print(f"Relevant count: {relevant_count}")
        #     print(f"Total relevant docs: {len(relevant_docs)}")
        #     print(f"Limit: {limit}")
        #     print()
        #     print()

        evaluation_results.append(
            {
                "query": query,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "retrieved": retrieved_titles,
                "relevant": relevant_docs,
            }
        )

    return evaluation_results

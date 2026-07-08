import argparse

from lib.augmented_search import (
    augmented_search,
    citations_search,
    question_answer,
    summarize_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Query")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        required=False,
        help="Number of results per search",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Add citations to the search result"
    )
    citations_parser.add_argument("query", type=str, help="Query")
    citations_parser.add_argument(
        "--limit", type=int, required=False, help="Limit", default=5
    )

    question_parser = subparsers.add_parser("question", help="Ask a question")
    question_parser.add_argument("query", type=str, help="Query")
    question_parser.add_argument(
        "--limit", type=int, default=5, required=False, help="Limit"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = augmented_search(query=query)
            print("Search Results:")
            for title in results["titles"]:
                print(f"- {title}")
            print(f"RAG Response:\n{results["llm_response"]}")
        case "summarize":
            query = args.query
            limit = args.limit
            response = summarize_search(query=query, k=limit)
            print("Search Results:")
            for title in response["titles"]:
                print(f"- {title}")
            print(f"LLM Summary:\n{response["llm_response"]}")
        case "citations":
            query = args.query
            limit = args.limit
            response = citations_search(query=query, limit=limit)
            print("Search Results:")
            for title in response["titles"]:
                print(f"- {title}")
            print(f"LLM Answer:\n{response["llm_response"]}")
        case "question":
            query = args.query
            limit = args.limit
            response = question_answer(question=query, limit=limit)
            print("Search Results:")
            for title in response["titles"]:
                print(f"- {title}")
            print(f"Answer:\n{response["llm_response"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

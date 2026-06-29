import argparse

from lib.hybrid_search import normalize_scores, rrf_search, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normaize_subparser = subparsers.add_parser(
        "normalize", help="Print normalized scores"
    )
    normaize_subparser.add_argument(
        "scores", nargs="*", help="List of scores separated by spaces"
    )

    weighted_search_subparser = subparsers.add_parser(
        "weighted-search", help="Weighted search"
    )
    weighted_search_subparser.add_argument("query", type=str, help="Query")
    weighted_search_subparser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=0.5,
        help="Alpha value. Default 0.5",
    )
    weighted_search_subparser.add_argument(
        "--limit", type=int, required=False, default=5, help="Limit. Default 5"
    )

    rrf_search_subparser = subparsers.add_parser("rrf-search", help="RRF Search")
    rrf_search_subparser.add_argument("query", type=str, help="Query")
    rrf_search_subparser.add_argument(
        "--k",
        type=int,
        required=False,
        default=60,
        help="K constant. Default 60",
    )
    rrf_search_subparser.add_argument(
        "--limit", type=int, required=False, default=5, help="Limit. Default 5"
    )
    rrf_search_subparser.add_argument(
        "--enhance",
        choices=["spell", "rewrite", "expand"],
        required=False,
        help="Query enhancement method",
    )
    rrf_search_subparser.add_argument(
        "--rerank-method",
        choices=["individual", "batch", "cross_encoder"],
        required=False,
        help="Re-Rank query",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = [float(score) for score in args.scores]
            normalized = normalize_scores(scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search(
                query=args.query, alpha=args.alpha, limit=args.limit
            )
            for index, result in enumerate(results, start=1):
                print(f"{index}.  {result["title"]}")
                print(f"  Hybrid Score: {result["hybrid_score"]:.3f}")
                print(
                    f"  BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}"
                )
                print(f"  {result["document"][:100]}")
        case "rrf-search":
            results = rrf_search(
                query=args.query,
                k=args.k,
                limit=args.limit,
                enhance_method=args.enhance,
                rerank_method=args.rerank_method,
            )
            print(
                f"Enhanced query ({args.enhance}): '{args.query}' -> '{results["enhanced_query"]}'\n"
            )
            for index, result in enumerate(results["results"], start=1):
                print(f"{index}.  {result["title"]}")
                if "rerank_score" in result:
                    print(f"  Re-rank Score: {result["rerank_score"]:.3f}/10")
                else:
                    print(f"  Re-rank Rank: {index}")
                if "cross_encoder_score" in result:
                    print(f"  Cross Encoder Score: {result["cross_encoder_score"]:.3f}")
                print(f"  RRF Score: {result["rrf_score"]:.3f}")
                print(
                    f"  BM25 Rank: {result["bm25_rank"]:.3f}, Semantic Rank: {result["semantic_rank"]:.3f}"
                )
                print(f"  {result["document"][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

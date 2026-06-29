import argparse

from lib.evaluation import evaluate_golden_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    eval_result = evaluate_golden_dataset(limit=limit)
    print(f"k={limit}")
    print()
    for result in eval_result:
        print(f"- Query: {result['query']}")
        print(f"  - Precision@{limit}: {result['precision']:.4f}")
        print(f"  - Recall@{limit}: {result['recall']:.4f}")
        print(f"  - F1 Score: {result['f1_score']:.4f}")
        print(f"  - Retrieved: {",".join(result['retrieved'])}")
        print(f"  - Relevant: {",".join(result['relevant'])}")
        print()


if __name__ == "__main__":
    main()

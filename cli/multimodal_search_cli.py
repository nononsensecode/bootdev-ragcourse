from argparse import ArgumentParser
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", description="Multimodal search commands")

    verify_img_embedding_subparser = subparsers.add_parser("verify_image_embedding", help="Verify image embeddings")
    verify_img_embedding_subparser.add_argument("image_path", type=str, help="Path to image")

    image_search_subparser = subparsers.add_parser("image_search", help="Search with image")
    image_search_subparser.add_argument("image", type=str, help="Path to image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image)
            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']} (similarity: {result['score']:.3f})\n  {result['description'][:100]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

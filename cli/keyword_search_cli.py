import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
from pathlib import Path
from collections import Counter, defaultdict


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.tokenizer = Tokenizer()

    def __add(self, doc_id: int, text: str) -> None:
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), []))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter())[term]

    def build(self):
        with open("data/movies.json") as file:
            movies = json.load(file)["movies"]
            for movie in movies:
                self.docmap[int(movie["id"])] = movie
                self.__add(int(movie["id"]), f"{movie["title"]} {movie["description"]}")

    def save(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
        except Exception as e:
            print(f"error occurred: {str(e)}")

        try:
            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
        except Exception as e:
            print(f"error occurred: {str(e)}")

        try:
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
        except Exception as e:
            print(f"error occurred: {str(e)}")


class Tokenizer:
    def __init__(self) -> None:
        with open("data/stop_words.txt", "r") as file:
            content = file.read()
            self.stop_words = content.splitlines()
        self.table = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()

    def tokenize(self, keywords: str) -> list[str]:
        tokens = []
        for token in keywords.split():
            if not token:
                continue
            token = self.stem(
                self._remove_punctuation(
                    self._remove_stop_words(self._make_case_insensitive(token.lower()))
                )
            )
            if token:
                tokens.append(token)
        return tokens

    def stem(self, word: str) -> str:
        return self.stemmer.stem(word, to_lowercase=True)

    def _remove_stop_words(self, word: str) -> str:
        return " ".join([w for w in word.split() if w not in self.stop_words])

    def _make_case_insensitive(self, word: str) -> str:
        return word.lower()

    def _remove_punctuation(self, word: str) -> str:
        return word.translate(self.table)


def tokenize_single_term(term: str) -> str:
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(term)
    if len(tokens) == 1:
        raise Exception("There should be exactly one token")
    return tokens[0]


def search(keyword: str) -> dict[int, str]:
    tokenizer = Tokenizer()
    inverted_index = InvertedIndex()
    inverted_index.load()
    if not inverted_index.index:
        raise Exception("index is not built!!")
    movies = {}
    words = sorted(tokenizer.tokenize(keyword))
    for word in words:
        doc_ids = inverted_index.get_documents(word)
        for doc_id in doc_ids:
            if len(movies) == 5:
                return movies
            movie = inverted_index.docmap[doc_id]
            movies[doc_id] = movie["title"]

    return movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build index and docmap")

    tf_parser = subparsers.add_parser("tf", help="Find term frequencies")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            movies = search(args.query)
            for id, movie in movies.items():
                print(f"{id}: {movie}")
        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            counts = inverted_index.term_frequencies.get(args.doc_id, Counter())
            return print(counts[args.term])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

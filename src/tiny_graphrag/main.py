import argparse
from argparse import Namespace
from pathlib import Path

from termcolor import colored

from tiny_graphrag.build import store_document
from tiny_graphrag.config import DB_URI
from tiny_graphrag.db import init_db
from tiny_graphrag.query import QueryEngine


def build_mode(args: Namespace) -> None:
    """Handle document processing and storage."""
    engine = init_db(DB_URI)
    print(f"Processing document: {args.input}")
    doc_id, graph_path = store_document(
        args.input, title=Path(args.input).stem, engine=engine
    )
    print(colored(f"Success! Document stored with ID: {doc_id}", "green"))
    print(colored(f"Graph saved to: {graph_path}", "green"))


def query_mode(args: Namespace) -> None:
    """Handle querying the stored document."""
    engine = init_db(DB_URI)
    query_engine = QueryEngine(engine)

    if args.mode == "local":
        if not args.graph:
            print("Error: --graph argument required for local search")
            return
        result = query_engine.local_search(args.query, args.graph)
        print("\nLocal Search Result:", result)

    elif args.mode == "global":
        if not args.doc_id:
            print("Error: --doc-id argument required for global search")
            return
        result = query_engine.global_search(args.query, args.doc_id)
        print("\nGlobal Search Result:", result)

    elif args.mode == "naive":
        result = query_engine.naive_search(args.query)
        print("\nNaive Search Result:", result)


def init_mode(_: Namespace) -> None:
    """Initialize the database."""
    init_db(DB_URI)
    print(colored("Database initialized successfully!", "green"))


def main() -> None:
    """Main driver."""
    parser = argparse.ArgumentParser(description="Tiny GraphRAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    subparsers.add_parser("init", help="Initialize the database")

    # Build command
    build_parser = subparsers.add_parser("build", help="Process and store a document")
    build_parser.add_argument("input", type=str, help="Path to input document")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query stored documents")
    query_parser.add_argument(
        "mode",
        choices=["local", "global", "naive"],
        help="Query mode: local (graph-based), global (vector-based), or naive (hybrid vector+keyword)",
    )
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument(
        "--graph",
        type=str,
        help="Path to graph pickle file (required for local search)",
    )
    query_parser.add_argument(
        "--doc-id", type=int, help="Document ID (required for global search)"
    )

    args = parser.parse_args()

    if args.command == "build":
        build_mode(args)
    elif args.command == "query":
        query_mode(args)
    elif args.command == "init":
        init_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

<p align="center">
  <img src=".github/logo.jpeg" alt="Tiny GraphRAG Logo" width="256"/>
</p>

# Tiny GraphRAG

A tiny 1000 line implementation of the GraphRAG algorithm using only language
models that run locally. This implementation is designed to be easy to be
easily understandable, hackable and extendable and not dependent on any
framework.

Notably this implementation does not use OpenAI or any commercial LLM providers
and can be configured to run locally on private data using only a MacBook Pro.

| Component         | Implementation                                    |
|------------------|--------------------------------------------------|
| Vector Database  | [pgvector](https://github.com/pgvector/pgvector) |
| Embedding Model  | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| Language Model   | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Meta-Llama-3.2-3B) |
| Entity Extractor | [gliner](https://github.com/urchade/GLiNER)     |
| Relation Extract | [glirel](https://github.com/jackboyla/GLiREL)   |
| Graph Database   | [networkx](https://github.com/networkx/networkx) |
| Inference        | [llama-cpp](https://github.com/abetlen/llama-cpp-python) |

## Usage

To install the dependencies run:

```bash
poetry install
```

To setup the local vector database create the docker container with:

```shell
docker-compose up -d
```

Then create the database tables with:

```shell
poetry run python db.py
```

To build the graph and embeddings for a document run:

```shell
poetry run python example build data/Barack_Obama.txt
```

To query use either the `local` or `global` mode and provide the path to the graph
file and the query:

```shell
# Local search
poetry run python example.py query local --graph graphs/21_graph.pkl "What did Barack Obama study at Columbia University?"

# Global search
poetry run python example.py query global --graph graphs/21_graph.pkl  --doc-id 21 "What are the main themes of this document?"
```

## Performance

The pipeline performance can be improved by using either Apple Silicon or a
Nvidia GPU to drive the embedding and language model inference. This will fallback
to CPU if no other options are available but the addition of dedicated hardware
will improve the performance.

**Apple Silicon**

```bash
poetry install --with apple
```

**Nvidia GPU**

```bash
poetry install --with nvidia
poetry run pip install flash-attn --no-build-isolation
```

## Library

You can also this library directly.

```python
from tiny_graphrag import store_document, QueryEngine

# Process and store a document
doc_id, graph_path = store_document(
    filepath="data/Barack_Obama.txt",
    title="Barack Obama Wikipedia",
)

query_engine = QueryEngine()

result = query_engine.local_search(
    query="What did Barack Obama study at Columbia University?",
    graph_path=graph_path
)

result = query_engine.global_search(
    query="What are the main themes of this document?",
    doc_id=doc_id
)
```

License
-------

MIT License

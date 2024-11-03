<p align="center">
  <img src=".github/logo.jpeg" alt="Tiny GraphRAG Logo" width="256"/>
</p>

# Tiny GraphRAG

A tiny 1000 line implementation of the GraphRAG algorithm using only language
models that run locally. This implementation is designed to be easy to be
easily understandable, hackable and extendable.

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

```bash
docker-compose up -d
```

Then create the database tables with:

```bash
poetry run python db.py
```

To run the example:

```bash
poetry run python example.py
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

License
-------

MIT License

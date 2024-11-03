# Tiny GraphRAG

<p align="center">
  <img src=".github/logo.jpeg" alt="Tiny GraphRAG Logo" width="500"/>
</p>

A tiny 1000 line implementation of the GraphRAG algorithm using only language
models that run locally. This implementation is designed to be easy to be
easily understandable, hackable and extendable.

| Component        | Implementation        |
|-----------------|----------------------|
| Vector Database | pgvector            |
| Embedding Model | sentence-transformers |
| Language Model  | meta-llama/Llama-3.2-3B |
| Entity Extractor| gliner              |
| Relation Extract| glirel              |
| Graph Database  | networkx            |
| Inference       | llama-cpp           |

License
-------

MIT License

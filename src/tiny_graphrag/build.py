from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import networkx as nx
from llama_cpp import Llama
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from tiny_graphrag.chunking import chunk_document
from tiny_graphrag.db import Document, DocumentChunk
from tiny_graphrag.dis import EntityDisambiguator
from tiny_graphrag.entity_types import MIN_ENTITY_TYPES
from tiny_graphrag.extract import extract_rels
from tiny_graphrag.graph import GraphStore
from tiny_graphrag.prompts import COMMUNITY_COMBINE, COMMUNITY_SUMMARY
from tiny_graphrag.rel_types import DEFAULT_RELS_LIST


@dataclass
class Entity:
    """Graph node."""

    id: str
    type: str


@dataclass
class Relation:
    """Graph triple."""

    head: str
    relation_type: str
    tail: str


@dataclass
class DocumentChunkData:
    """Document chunk data."""

    text: str
    embedding: Any


@dataclass
class ProcessedDocument:
    """Processed document data."""

    chunks: List[DocumentChunkData]
    entities: List[Entity]
    relations: List[Relation]


def process_document(
    filepath: str,
    title: Optional[str] = None,
    max_chunks: int = -1,
    entity_types: List[str] = MIN_ENTITY_TYPES,
    relation_types: List[str] = DEFAULT_RELS_LIST,
) -> ProcessedDocument:
    """Process a document and return chunks, entities and relations."""
    # Read and chunk document
    page_text = open(filepath).read()
    page_chunks = chunk_document(page_text)

    # Initialize disambiguator
    disambiguator = EntityDisambiguator()

    # Build graph and collect entities/relations
    g = nx.Graph()
    entities: List[Entity] = []
    relations: List[Relation] = []
    chunks: List[DocumentChunkData] = []

    # First pass: collect all entity mentions with context
    all_mentions = []
    for chunk_text, embedding in tqdm(page_chunks[:max_chunks]):
        chunks.append(DocumentChunkData(text=chunk_text, embedding=embedding))
        extraction = extract_rels(chunk_text, entity_types, relation_types)

        # Add entities from both standalone entities and those in relations
        for ent in extraction.entities:
            all_mentions.append((ent[0], ent[1], chunk_text))
        # Add entities from relations to ensure they're all captured
        for rel in extraction.relations:
            all_mentions.append((rel[0], "UNKNOWN", chunk_text))
            all_mentions.append((rel[2], "UNKNOWN", chunk_text))

    # Remove duplicates while preserving order
    all_mentions = list(dict.fromkeys(tuple(m) for m in all_mentions))

    # Resolve entities to canonical forms
    resolved_entities = disambiguator.resolve_entities(all_mentions)
    entity_map = {m[0]: r[0] for m, r in zip(all_mentions, resolved_entities)}

    # Second pass: create graph with resolved entities
    for chunk_text, _ in page_chunks[:max_chunks]:
        extraction = extract_rels(chunk_text, entity_types, relation_types)

        for ent in extraction.entities:
            if ent[0] in entity_map:  # Only add if entity was resolved
                canonical_form = entity_map[ent[0]]
                g.add_node(canonical_form, label=ent[1])
                entities.append(Entity(id=canonical_form, type=ent[1]))

        for rel in extraction.relations:
            # Skip relations where either entity wasn't resolved
            if rel[0] not in entity_map or rel[2] not in entity_map:
                continue

            head_canonical = entity_map[rel[0]]
            tail_canonical = entity_map[rel[2]]
            g.add_edge(
                head_canonical, tail_canonical, label=rel[1], source_chunk=chunk_text
            )
            relations.append(
                Relation(head=head_canonical, relation_type=rel[1], tail=tail_canonical)
            )

    return ProcessedDocument(chunks=chunks, entities=entities, relations=relations)


def generate_community_summary(
    llm: Llama,
    community: List[Tuple[str, str, str, str]],
    max_triples: int = 30,
    temperature: float = 0.2,
) -> str:
    """Generate a summary for a community, chunking if needed."""
    # Chunk the community into smaller pieces if too large
    chunks = [
        community[i : i + max_triples] for i in range(0, len(community), max_triples)
    ]

    summaries: List[str] = []
    for chunk in chunks:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": COMMUNITY_SUMMARY.format(community=chunk),
                }
            ],
            temperature=temperature,
        )
        summaries.append(response["choices"][0]["message"]["content"])  # type: ignore

    # If we had multiple chunks, combine them
    if len(summaries) > 1:
        combined = " ".join(summaries)
        # Generate a final summary of the combined text
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": COMMUNITY_COMBINE.format(combined=combined),
                }
            ],
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]  # type: ignore

    return summaries[0]


def store_document(
    filepath: str,
    engine: Engine,
    title: Optional[str] = None,
    max_chunks: int = -1,
) -> int:
    """Store document in database and graph store."""
    session_local = sessionmaker(bind=engine)
    graph_store = GraphStore()

    try:
        with session_local() as session:
            # Process document
            processed = process_document(filepath, title, max_chunks)

            # Store document
            doc = Document(content=open(filepath).read(), title=title)
            session.add(doc)
            session.flush()

            # Store chunks with embeddings
            for chunk in processed.chunks:
                chunk_obj = DocumentChunk(
                    document_id=doc.id,
                    content=chunk.text,
                    embedding=chunk.embedding,
                    chunk_index=0,
                )
                session.add(chunk_obj)

            # Store in graph database
            graph_store.store_graph(
                doc.id,
                [(e.id, e.type) for e in processed.entities],
                [(r.head, r.relation_type, r.tail) for r in processed.relations],
                processed.chunks[0].text,
            )

            session.commit()
            return doc.id
    finally:
        graph_store.close()

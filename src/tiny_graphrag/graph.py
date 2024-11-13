from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase


@dataclass
class GraphConfig:
    """Memgraph connection configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "admin"
    password: str = "admin"
    database: str = "memgraph"


class GraphStore:
    """Handles graph storage and retrieval using Memgraph."""

    def __init__(self, config: GraphConfig | None = None):
        """Initialize connection to Memgraph."""
        self.driver = GraphDatabase.driver(
            (config or GraphConfig()).uri,
            auth=(
                (
                    (config or GraphConfig()).username,
                    (config or GraphConfig()).password,
                )
                if (config or GraphConfig()).username
                else None
            ),
        )

    def close(self) -> None:
        """Close the driver connection."""
        self.driver.close()

    def store_graph(
        self,
        doc_id: int,
        entities: List[Tuple[str, str]],
        relations: List[Tuple[str, str, str]],
        source_chunk: str,
    ) -> None:
        """Store entities and relations in Memgraph."""
        with self.driver.session() as session:
            # Store entities
            for text, label in entities:
                query = "MERGE (n:Entity {content: $content, label: $label, doc_id: $doc_id})"
                session.run(query, {"content": text, "label": label, "doc_id": doc_id})

            # Store relations with properties
            for head, rel_type, tail in relations:
                query = """
                    MATCH (h:Entity {content: $head}), (t:Entity {content: $tail})
                    CREATE (h)-[:RELATES {type: $rel_type, doc_id: $doc_id, source_chunk: $source_chunk}]->(t)
                """
                session.run(
                    query,
                    {
                        "head": head,
                        "tail": tail,
                        "rel_type": rel_type,
                        "doc_id": doc_id,
                        "source_chunk": source_chunk,
                    },
                )

    def get_subgraph(self, doc_id: int) -> List[Dict[str, Any]]:
        """Retrieve all entities and relations for a document."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES]->(t:Entity)
                WHERE r.doc_id = $doc_id
                RETURN h.text as head_text, h.label as head_label,
                       r.type as rel_type, r.source_chunk as chunk,
                       t.text as tail_text, t.label as tail_label
                """,
                {"doc_id": doc_id},
            )
            return result.data()

    def get_relevant_data(
        self, doc_id: int, query_entities: List[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant entities and relations for query entities."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES]->(t:Entity)
                WHERE r.doc_id = $doc_id
                AND (toLower(h.text) CONTAINS toLower($query) OR toLower(t.text) CONTAINS toLower($query))
                RETURN h.text as head_text, h.label as head_label,
                       r.type as rel_type, r.source_chunk as chunk,
                       t.text as tail_text, t.label as tail_label
                """,
                {"doc_id": doc_id, "query": " OR ".join(query_entities)},
            )
            return result.data()

from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from typing import List
from db import engine


@dataclass
class SearchResult:
    """Represents a single search result with its relevance score."""

    document_id: int
    content: str
    score: float


# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
model = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_search(query: str, limit: int = 5, k: int = 60) -> List[SearchResult]:
    """
    Perform hybrid search combining semantic and keyword search.
    """
    # Generate embedding for the query
    query_embedding = model.encode(query)

    # SQL for hybrid search using RRF (Reciprocal Rank Fusion)
    sql = text(
        """
    WITH semantic_search AS (
        SELECT id, content, document_id,
               RANK () OVER (ORDER BY embedding <=> (:embedding)::vector) AS rank
        FROM document_chunks
        ORDER BY embedding <=> (:embedding)::vector
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, content, document_id,
               RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC)
        FROM document_chunks, plainto_tsquery('english', :query) query
        WHERE to_tsvector('english', content) @@ query
        ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(semantic_search.document_id, keyword_search.document_id) AS document_id,
        COALESCE(semantic_search.content, keyword_search.content) AS content,
        COALESCE(1.0 / (:k + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (:k + keyword_search.rank), 0.0) AS score
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
    ORDER BY score DESC
    LIMIT :limit
    """
    )

    # Create a sessionmaker
    SessionLocal = sessionmaker(bind=engine)

    # Use the session
    session = SessionLocal()
    try:
        results = session.execute(
            sql,
            {
                "query": query,
                "embedding": query_embedding.tolist(),
                "k": k,
                "limit": limit,
            },
        ).fetchall()

        return [
            SearchResult(
                document_id=row.document_id, content=row.content, score=row.score
            )
            for row in results
        ]
    finally:
        session.close()


# Implmenet an example search
if __name__ == "__main__":
    results = hybrid_search("the color white")
    for result in results:
        print(
            f"Document ID: {result.document_id}, Content: {result.content}, Score: {result.score}"
        )
        print("-" * 100)

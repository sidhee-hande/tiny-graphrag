from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (  # type: ignore
    DeclarativeBase,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func

from tiny_graphrag.config import DB_URI


class Base(DeclarativeBase):  # type: ignore
    """Base class for all SQLAlchemy ORM models in the application."""

    pass


class Document(Base):
    """Represents a document in the database.

    Stores the original document content and metadata.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    title = Column(String(256))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    communities = relationship(
        "Community", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Represents a chunk of a document in the database.

    Stores smaller segments of documents with their embeddings for efficient search.
    """

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = mapped_column(Vector(384), nullable=False)
    page_number = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

    # Indexes
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_document_chunks_content_fts",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
    )

    # Relationships
    document = relationship("Document", back_populates="chunks")


class Summary(Base):
    """Represents a document summary in the database.

    Stores generated summaries of documents.
    """

    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding = mapped_column(Vector(384), nullable=False)

    # Indexes
    __table_args__ = (
        Index(
            "ix_summaries_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_summaries_content_fts",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
    )


class Community(Base):
    """Represents a community of related documents in the database.

    Stores information about document clusters and their relationships.
    """

    __tablename__ = "communities"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)  # Summary text
    embedding = mapped_column(Vector(384), nullable=False)
    nodes = Column(Text, nullable=False)  # JSON string of node IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="communities")

    # Indexes
    __table_args__ = (
        Index(
            "ix_communities_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# Database connection setup
def init_db(db_uri: str) -> Engine:
    """Initialize the database connection and create necessary tables.

    Returns:
        Engine: SQLAlchemy engine instance connected to the database.
    """
    engine = create_engine(db_uri)

    # Enable pgvector extension if it's not already enabled
    print("Enabling pgvector extension...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    print("Creating tables...")
    Base.metadata.create_all(engine)

    print("Creating indexes...")
    # Create all of the indexes for each model
    for model in [Document, DocumentChunk, Summary, Community]:
        for index in getattr(model, "__table_args__", []):
            if isinstance(index, Index):
                try:
                    index.create(engine)
                except Exception:
                    print(f"Index already exists, skipping: {index.name}")

    print("Done.")
    return engine


if __name__ == "__main__":
    init_db(DB_URI)

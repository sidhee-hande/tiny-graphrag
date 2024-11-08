from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EntityMention:
    """Represents a mention of an entity in text."""

    text: str  # The entity text
    type: str  # Entity type
    context: str  # Surrounding text context
    embedding: np.ndarray  # Vector embedding


class EntityDisambiguator:
    """Handles entity disambiguation and coreference resolution."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the disambiguator.

        Args:
            similarity_threshold: Minimum cosine similarity for entity matching
        """
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold
        self.entity_clusters: Dict[str, Set[str]] = (
            {}
        )  # Maps canonical forms to variants
        self.mention_embeddings: Dict[str, np.ndarray] = {}

    def get_context_embedding(self, mention: str, context: str) -> np.ndarray:
        """Generate embedding for entity mention with context."""
        text_to_encode = f"{mention} | {context}"
        return self.model.encode(text_to_encode)

    def are_same_entity(self, mention1: EntityMention, mention2: EntityMention) -> bool:
        """Determine if two mentions refer to the same entity."""
        # Check if entities are of same type
        if mention1.type != mention2.type:
            return False

        # Compare embeddings using cosine similarity
        similarity = np.dot(mention1.embedding, mention2.embedding) / (
            np.linalg.norm(mention1.embedding) * np.linalg.norm(mention2.embedding)
        )
        return similarity >= self.similarity_threshold

    def get_canonical_form(self, mention: EntityMention) -> str:
        """Get canonical form for an entity mention."""
        for canonical, _variants in self.entity_clusters.items():
            canonical_embedding = self.mention_embeddings[canonical]
            similarity = np.dot(mention.embedding, canonical_embedding) / (
                np.linalg.norm(mention.embedding) * np.linalg.norm(canonical_embedding)
            )
            if similarity >= self.similarity_threshold:
                return canonical
        return mention.text

    def add_mention(self, mention: EntityMention) -> str:
        """Add a new entity mention and return its canonical form."""
        canonical_form = self.get_canonical_form(mention)

        if canonical_form == mention.text:
            # This is a new entity cluster
            self.entity_clusters[mention.text] = {mention.text}
            self.mention_embeddings[mention.text] = mention.embedding
        else:
            # Add to existing cluster
            self.entity_clusters[canonical_form].add(mention.text)

        return canonical_form

    def resolve_entities(
        self, mentions: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str]]:
        """Resolve a list of entity mentions to their canonical forms.

        Args:
            mentions: List of (entity_text, entity_type, context) tuples

        Returns:
            List of (canonical_form, entity_type) tuples
        """
        resolved_entities = []

        for text, type_, context in mentions:
            embedding = self.get_context_embedding(text, context)
            mention = EntityMention(text, type_, context, embedding)
            canonical_form = self.add_mention(mention)
            resolved_entities.append((canonical_form, type_))

        return resolved_entities

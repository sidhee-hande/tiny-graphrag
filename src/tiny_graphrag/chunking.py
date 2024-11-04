from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def chunk_document(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> List[Tuple[str, npt.NDArray[np.float32]]]:
    """Split a document into overlapping chunks and generate embeddings for each chunk.

    Args:
        text: The input document text
        chunk_size: Approximate size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of tuples containing (chunk_text, embedding_vector)
    """
    # Split text into sentences (simple split on periods for now)
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    # Build chunks from sentences
    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            # Join the current chunk sentences and get embedding
            chunk_text = ". ".join(current_chunk) + "."
            # Convert tensor to numpy array
            embedding = model.encode(
                chunk_text, convert_to_numpy=True, show_progress_bar=False
            )
            chunks.append((chunk_text, embedding))

            # Start new chunk with overlap
            # Keep last few sentences for overlap
            overlap_size = 0
            overlap_chunk: List[str] = []
            for s in reversed(current_chunk):
                s_len = len(s)
                if overlap_size + s_len <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += s_len
                else:
                    break

            current_chunk = overlap_chunk
            current_length = overlap_size

        current_chunk.append(sentence)
        current_length += sentence_length

    # Handle the last chunk
    if current_chunk:
        chunk_text = ". ".join(current_chunk) + "."
        # Convert tensor to numpy array
        embedding = model.encode(
            chunk_text, convert_to_numpy=True, show_progress_bar=False
        )
        chunks.append((chunk_text, embedding))

    return chunks


def get_embedding_dim() -> int:
    """Get the dimensionality of the embedding model."""
    return cast(int, model.get_sentence_embedding_dimension())

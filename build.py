from typing import List, Tuple
import pickle
import networkx as nx
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import json
from llama_cpp import Llama
from prompts import COMMUNITY_SUMMARY

from chunking import chunk_document, model
from communities import build_communities
from extract import extract_rels
from visualize import visualize, visualize_communities
from db import Document, DocumentChunk, init_db, engine, Community
from config import MODEL_REPO, MODEL_ID


def process_document(
    filepath: str, title: str = None, max_chunks: int = -1
) -> Tuple[List[Tuple[str, any]], nx.Graph]:
    """Process a document and return chunks and graph"""
    # Read and chunk document
    page_text = open(filepath).read()
    page_chunks = chunk_document(page_text)

    # Build graph
    g = nx.Graph()

    for chunk_text, embedding in tqdm(page_chunks[:max_chunks]):
        ents, rels = extract_rels(chunk_text)

        for ent in ents:
            g.add_node(ent[0], label=ent[1])

        for rel in rels:
            g.add_edge(rel[0], rel[2], label=rel[1], source_chunk=chunk_text)

    return page_chunks, g


def generate_community_summary(llm, community, max_triples=30):
    """Generate a summary for a community, chunking if needed"""
    # Chunk the community into smaller pieces if too large
    chunks = [
        community[i : i + max_triples] for i in range(0, len(community), max_triples)
    ]

    summaries = []
    for chunk in chunks:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": COMMUNITY_SUMMARY.format(community=chunk),
                }
            ],
            temperature=0.2,
        )
        summaries.append(response["choices"][0]["message"]["content"])

    # If we had multiple chunks, combine them
    if len(summaries) > 1:
        combined = " ".join(summaries)
        # Generate a final summary of the combined text
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize these related points into a single coherent paragraph: {combined}",
                }
            ],
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"]

    return summaries[0]


def store_document(filepath: str, title: str = None, max_chunks: int = -1):
    """Store document in database and save graph"""
    # Initialize database and LLM
    init_db()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO, filename=MODEL_ID, local_dir=".", verbose=False, n_ctx=4096
    )

    try:
        # Process document
        chunks, graph = process_document(filepath, title, max_chunks)

        # Store document
        doc = Document(content=open(filepath).read(), title=title or filepath)
        session.add(doc)
        session.flush()  # Get document ID

        # Store chunks
        for chunk_text, embedding in chunks:
            chunk = DocumentChunk(
                document_id=doc.id,
                content=chunk_text,
                embedding=embedding,
                chunk_index=0,
            )
            session.add(chunk)

        # Build and store communities
        communities, community_map = build_communities(graph)

        # Generate and store community summaries
        for community in communities:
            # Generate summary with chunking
            summary = generate_community_summary(llm, community)

            # Get embedding for summary
            embedding = model.encode(
                summary, convert_to_numpy=True, show_progress_bar=False
            )

            # Store community
            community_obj = Community(
                document_id=doc.id,
                content=summary,
                embedding=embedding,
                nodes=json.dumps(
                    [
                        n
                        for n, c in community_map.items()
                        if c == communities.index(community)
                    ]
                ),
            )
            session.add(community_obj)

        # Save graph
        graph_path = f"graphs/{doc.id}_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)

        # Visualize graphs
        visualize(graph, f"graphs/{doc.id}_graph.png")
        visualize_communities(graph, communities, f"graphs/{doc.id}_communities.png")

        session.commit()
        return doc.id, graph_path

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    doc_id, graph_path = store_document("data/Barack_Obama.txt", "Barack Obama")
    print(f"Stored document {doc_id}, graph saved to {graph_path}")

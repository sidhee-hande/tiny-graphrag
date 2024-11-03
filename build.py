from typing import List, Tuple
import pickle
import networkx as nx
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from chunking import chunk_document
from extract import extract_rels
from visualize import visualize
from db import Document, DocumentChunk, init_db, engine

def process_document(filepath: str, title: str = None) -> Tuple[List[Tuple[str, any]], nx.Graph]:
    """Process a document and return chunks and graph"""
    # Read and chunk document
    page_text = open(filepath).read()
    page_chunks = chunk_document(page_text)
    
    # Build graph
    g = nx.Graph()
    
    for chunk_text, embedding in tqdm(page_chunks):
        ents, rels = extract_rels(chunk_text)
        
        for ent in ents:
            g.add_node(ent[0], label=ent[1])
            
        for rel in rels:
            g.add_edge(rel[0], rel[2], label=rel[1], source_chunk=chunk_text)
    
    return page_chunks, g

def store_document(filepath: str, title: str = None):
    """Store document in database and save graph"""
    # Initialize database
    init_db()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Process document
        chunks, graph = process_document(filepath, title)
        
        # Store document
        doc = Document(
            content=open(filepath).read(),
            title=title or filepath
        )
        session.add(doc)
        session.flush()  # Get document ID
        
        # Store chunks
        for chunk_text, embedding in chunks:
            chunk = DocumentChunk(
                document_id=doc.id,
                content=chunk_text,
                embedding=embedding,
                chunk_index=0  # You may want to track proper indices
            )
            session.add(chunk)
        
        # Save graph
        graph_path = f"graphs/{doc.id}_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
            
        # Visualize graph
        visualize(graph, f"graphs/{doc.id}_graph.png")
        
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

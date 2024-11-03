import pickle
import networkx as nx
from llama_cpp import Llama
from sqlalchemy.orm import sessionmaker

from search import hybrid_search
from config import MODEL_REPO, MODEL_ID
from db import engine
from prompts import LOCAL_SEARCH


class QueryEngine:
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_ID,
            local_dir=".",
            verbose=False,
            n_ctx=2048,
        )
        self.SessionLocal = sessionmaker(bind=engine)

    def load_graph(self, graph_path: str) -> nx.Graph:
        """Load graph from pickle file"""
        with open(graph_path, "rb") as f:
            return pickle.load(f)

    def local_search(self, query: str, graph_path: str) -> str:
        """Perform local search using graph"""
        g = self.load_graph(graph_path)

        # Extract entities from query
        query_response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": LOCAL_SEARCH.format(query=query)}],
            temperature=0.1,
        )
        query_entities = query_response["choices"][0]["message"]["content"].split("\n")
        query_entities = [e.strip() for e in query_entities]

        # Find relevant information
        relevant_data = {"entities": set(), "relationships": [], "text_chunks": set()}

        for query_entity in query_entities:
            for node in g.nodes():
                if query_entity.lower() in str(node).lower():
                    relevant_data["entities"].add(node)

                    for neighbor in g.neighbors(node):
                        relevant_data["entities"].add(neighbor)
                        edge_data = g.get_edge_data(node, neighbor)
                        if edge_data:
                            rel = (node, edge_data.get("label", ""), neighbor)
                            relevant_data["relationships"].append(rel)
                            if "source_chunk" in edge_data:
                                relevant_data["text_chunks"].add(
                                    edge_data["source_chunk"]
                                )

        # Build context and generate response
        context = self._build_context(relevant_data)
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Answer the query using only the provided context. Be specific and concise.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.3,
        )

        return response["choices"][0]["message"]["content"]

    def global_search(self, query: str, doc_id: int, limit: int = 5) -> str:
        """Perform global search using vector database"""
        results = hybrid_search(query, limit=limit)

        # Combine results into final response
        context = "\n\n".join([r.content for r in results])
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Answer the query based on the provided context.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.3,
        )

        return response["choices"][0]["message"]["content"]

    def _build_context(self, relevant_data: dict) -> str:
        return f"""
        Relevant Entities: {', '.join(str(e) for e in relevant_data['entities'])}
        
        Relationships:
        {'\n'.join(f'{s} {r} {t}' for s, r, t in relevant_data['relationships'])}
        
        Supporting Text:
        {'\n'.join(relevant_data['text_chunks'])}
        """


if __name__ == "__main__":
    engine = QueryEngine()

    # Example queries
    local_result = engine.local_search(
        "What did Barack Obama study at Columbia University?", "graphs/1_graph.pkl"
    )
    print("\nLocal Search Result:", local_result)

    global_result = engine.global_search(
        "What were Obama's major achievements as president?", doc_id=1
    )
    print("\nGlobal Search Result:", global_result)

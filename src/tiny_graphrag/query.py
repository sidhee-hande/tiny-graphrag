import pickle
import networkx as nx
from llama_cpp import Llama
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Set, List, Tuple

from tiny_graphrag.config import MODEL_REPO, MODEL_ID
from tiny_graphrag.db import engine
from tiny_graphrag.prompts import (
    LOCAL_SEARCH,
    LOCAL_SEARCH_RESPONSE,
    GLOBAL_SEARCH_COMMUNITY,
    GLOBAL_SEARCH_COMBINE,
    NAIVE_SEARCH_RESPONSE,
)
from tiny_graphrag.db import Community


@dataclass
class RelevantData:
    entities: Set[str] = field(default_factory=set)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    text_chunks: Set[str] = field(default_factory=set)


class QueryEngine:
    def __init__(self) -> None:
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
        query_entities = query_response["choices"][0]["message"]["content"].split("\n")  # type: ignore
        query_entities = [e.strip() for e in query_entities]

        # Replace dictionary with dataclass
        relevant_data = RelevantData()

        for query_entity in query_entities:
            for node in g.nodes():
                if query_entity.lower() in str(node).lower():
                    relevant_data.entities.add(node)

                    for neighbor in g.neighbors(node):
                        relevant_data.entities.add(neighbor)
                        edge_data = g.get_edge_data(node, neighbor)
                        if edge_data:
                            rel = (node, edge_data.get("label", ""), neighbor)
                            relevant_data.relationships.append(rel)
                            if "source_chunk" in edge_data:
                                relevant_data.text_chunks.add(
                                    edge_data["source_chunk"]
                                )

        # Build context and generate response
        context = self._build_context(relevant_data)
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": LOCAL_SEARCH_RESPONSE,
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.3,
        )

        return response["choices"][0]["message"]["content"]  # type: ignore

    def global_search(self, query: str, doc_id: int, limit: int = 5) -> str:
        """Perform global search using community summaries and vector database"""
        # Create session
        session = self.SessionLocal()
        try:
            # Get all community summaries for the document
            communities = (
                session.query(Community).filter(Community.document_id == doc_id).all()
            )

            # Map phase - get answers from each community
            intermediate_answers = []
            print("Performing map phase over communities.")

            for community in tqdm(communities):
                response = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": GLOBAL_SEARCH_COMMUNITY,
                        },
                        {
                            "role": "user",
                            "content": f"Community Summary:\n{community.content}\n\nQuery: {query}",
                        },
                    ],
                    temperature=0.7,
                )
                answer = response["choices"][0]["message"]["content"]  # type: ignore
                if answer and "No relevant information found" not in answer:
                    intermediate_answers.append(answer)

            # Reduce phase - combine community answers
            print("Performing reduce phase over community answers.")
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": GLOBAL_SEARCH_COMBINE,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nAnswers to combine:\n{' '.join(intermediate_answers)}",
                    },
                ],
                temperature=0.7,
            )

            return response["choices"][0]["message"]["content"]  # type: ignore

        finally:
            session.close()

    def _build_context(self, relevant_data: RelevantData) -> str:
        return f"""
        Relevant Entities: {', '.join(str(e) for e in relevant_data.entities)}
        
        Relationships:
        {'\n'.join(f'{s} {r} {t}' for s, r, t in relevant_data.relationships)}
        
        Supporting Text:
        {'\n'.join(relevant_data.text_chunks)}
        """

    def naive_search(self, query: str, limit: int = 5) -> str:
        """Perform naive RAG search using hybrid vector + keyword search"""
        from tiny_graphrag.search import hybrid_search

        # Get relevant chunks using hybrid search
        search_results = hybrid_search(query, limit=limit)

        # Build context from search results
        context = "\n\n".join([result.content for result in search_results])

        # Generate response using LLM
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": NAIVE_SEARCH_RESPONSE,
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.3,
        )

        return response["choices"][0]["message"]["content"]  # type: ignore
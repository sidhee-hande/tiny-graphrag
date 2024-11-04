import pickle
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import networkx as nx
from llama_cpp import Llama
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from tiny_graphrag.config import MODEL_ID, MODEL_REPO
from tiny_graphrag.db import Community
from tiny_graphrag.prompts import (
    GLOBAL_SEARCH_COMBINE,
    GLOBAL_SEARCH_COMMUNITY,
    LOCAL_SEARCH,
    LOCAL_SEARCH_CONTEXT,
    LOCAL_SEARCH_RESPONSE,
    NAIVE_SEARCH_RESPONSE,
)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_LIMIT = 10
DEFAULT_CTX_LENGTH = 8192


@dataclass
class RelevantData:
    """Container for data retrieved during local graph search."""

    entities: Set[str] = field(default_factory=set)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    text_chunks: Set[str] = field(default_factory=set)


class QueryEngine:
    """Engine for performing various types of semantic searches."""

    def __init__(self, engine: Engine) -> None:
        """Initialize QueryEngine with LLM model and database session."""
        self.llm = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_ID,
            local_dir=".",
            verbose=False,
            n_ctx=DEFAULT_CTX_LENGTH,
        )
        self.SessionLocal = sessionmaker(bind=engine)
        self.engine = engine

    def load_graph(self, graph_path: str) -> nx.Graph:
        """Load graph from pickle file.

        Args:
            graph_path: Path to the pickled graph file.

        Returns:
            NetworkX graph object.

        Raises:
            FileNotFoundError: If graph file doesn't exist.
        """
        try:
            with open(graph_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Graph file not found at: {graph_path}") from err

    def local_search(self, query: str, graph_path: str) -> str:
        """Perform local search using graph structure.

        Args:
            query: User query string.
            graph_path: Path to the graph file.

        Returns:
            Generated response based on local graph search.
        """
        g = self.load_graph(graph_path)

        # Extract entities from query
        query_entities = self._extract_query_entities(query)
        relevant_data = self._gather_relevant_data(g, query_entities)

        # Build context and generate response
        context = self._build_context(relevant_data)
        return self._generate_response(query, context)

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract relevant entities from the query using LLM."""
        response = self._generate_llm_response(
            "", LOCAL_SEARCH.format(query=query), temp=0.1
        )
        return [e.strip() for e in response.split("\n")]

    def _gather_relevant_data(
        self, graph: nx.Graph, query_entities: List[str]
    ) -> RelevantData:
        """Gather relevant nodes, relationships and text chunks from graph."""
        relevant_data = RelevantData()

        for query_entity in query_entities:
            for node in graph.nodes():
                if query_entity.lower() in str(node).lower():
                    self._process_matching_node(node, graph, relevant_data)

        return relevant_data

    def _process_matching_node(
        self, node: str, graph: nx.Graph, relevant_data: RelevantData
    ) -> None:
        """Process a matching node and its neighbors."""
        relevant_data.entities.add(node)

        for neighbor in graph.neighbors(node):
            relevant_data.entities.add(neighbor)
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data:
                rel = (node, edge_data.get("label", ""), neighbor)
                relevant_data.relationships.append(rel)
                if "source_chunk" in edge_data:
                    relevant_data.text_chunks.add(edge_data["source_chunk"])

    def global_search(self, query: str, doc_id: int, limit: int = 5) -> str:
        """Perform global search using community summaries and vector database."""
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
                answer = self._generate_llm_response(
                    GLOBAL_SEARCH_COMMUNITY,
                    f"Community Summary:\n{community.content}\n\nQuery: {query}",
                    temp=0.7,
                )
                if answer and "No relevant information found" not in answer:
                    intermediate_answers.append(answer)

            # Reduce phase - combine community answers
            print("Performing reduce phase over community answers.")
            return self._generate_llm_response(
                GLOBAL_SEARCH_COMBINE,
                f"Query: {query}\n\nAnswers to combine:\n{' '.join(intermediate_answers)}",
                temp=0.7,
            )

        finally:
            session.close()

    def _build_context(self, relevant_data: RelevantData) -> str:
        """Build context string from relevant data for LLM prompt."""
        entities_str = ", ".join(str(e) for e in relevant_data.entities)
        relationships_str = "\n".join(
            f"{s} {r} {t}" for s, r, t in relevant_data.relationships
        )
        text_chunks_str = "\n".join(relevant_data.text_chunks)

        return LOCAL_SEARCH_CONTEXT.format(
            entities=entities_str,
            relationships=relationships_str,
            text_chunks=text_chunks_str,
        )

    def naive_search(self, query: str, limit: int = 5) -> str:
        """Perform naive RAG search using hybrid vector + keyword search."""
        from tiny_graphrag.search import hybrid_search

        # Get relevant chunks using hybrid search
        search_results = hybrid_search(query, limit=limit, engine=self.engine)

        # Build context from search results
        context = "\n\n".join([result.content for result in search_results])

        print(f"Context:\n{context}")

        # Generate response using LLM
        return self._generate_llm_response(
            NAIVE_SEARCH_RESPONSE, f"Context:\n{context}\n\nQuery: {query}"
        )

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM based on query and context.

        Args:
            query: User query string.
            context: Context information for the LLM.

        Returns:
            Generated response string.
        """
        return self._generate_llm_response(
            LOCAL_SEARCH_RESPONSE, f"Context:\n{context}\n\nQuery: {query}"
        )

    def _generate_llm_response(
        self, system_prompt: str, user_content: str, temp: float = DEFAULT_TEMPERATURE
    ) -> str:
        """Common method for generating LLM responses.

        Args:
            system_prompt: System prompt for the LLM.
            user_content: User content/query.
            temp: Temperature parameter for generation.

        Returns:
            Generated response string.
        """
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temp,
        )

        if not response or "choices" not in response:
            raise RuntimeError("Failed to get valid response from LLM")

        return str(response["choices"][0]["message"]["content"])  # type: ignore

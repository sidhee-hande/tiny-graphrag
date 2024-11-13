from dataclasses import dataclass, field
from typing import List, Set, Tuple

from llama_cpp import Llama
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from tiny_graphrag.config import MODEL_ID, MODEL_REPO
from tiny_graphrag.db import Community
from tiny_graphrag.graph import GraphStore
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
        self.graph_store = GraphStore()

    def local_search(self, query: str, doc_id: int) -> str:
        """Perform local search using graph structure."""
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)

            # Get relevant data from graph store and convert to RelevantData
            graph_data = self.graph_store.get_relevant_data(doc_id, query_entities)

            # Initialize RelevantData object
            relevant_data = RelevantData()

            # Populate RelevantData from graph_data
            for item in graph_data:
                if "entity" in item:
                    relevant_data.entities.add(item["entity"])
                if "head" in item and "relation" in item and "tail" in item:
                    relevant_data.relationships.append(
                        (item["head"], item["relation"], item["tail"])
                    )
                if "text_chunk" in item:
                    relevant_data.text_chunks.add(item["text_chunk"])

            # Build context and generate response
            context = self._build_context(relevant_data)
            return self._generate_response(query, context)
        finally:
            self.graph_store.close()

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract relevant entities from the query using LLM."""
        response = self._generate_llm_response(
            "", LOCAL_SEARCH.format(query=query), temp=0.1
        )
        return [e.strip() for e in response.split("\n")]

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

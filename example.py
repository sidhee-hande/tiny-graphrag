from cdlib import algorithms
import networkx as nx
from llama_cpp import Llama

from chunk import chunk_document, model
from extract import extract_rels
from visualize import visualize, visualize_communities


def build_chunks(ifile):
    page_text = open(ifile).read()
    page_chunks = chunk_document(page_text)
    return page_chunks


def build_graph(page_chunks, max_chunks=-1):
    g = nx.Graph()

    for chunk_text, embedding in page_chunks[:max_chunks]:
        ents, rels = extract_rels(chunk_text)

        for ent in ents:
            g.add_node(ent[0], label=ent[1])

        for rel in rels:
            g.add_edge(rel[0], rel[2], label=rel[1], source_chunk=chunk_text)

    return g


def build_communities(g):
    communities = []
    node_community_map = {}

    # Convert to integers while preserving all node attributes
    gi = nx.convert_node_labels_to_integers(g, label_attribute="original_label")

    # Create reverse mapping for nodes
    reverse_mapping = {
        node: data["original_label"] for node, data in gi.nodes(data=True)
    }

    for component in nx.connected_components(gi):
        subgraph = gi.subgraph(component)
        if len(subgraph) > 1:
            coms = algorithms.leiden(subgraph)
            for com_id, com in enumerate(coms.communities):
                # Get the subgraph for this community
                community_subgraph = coms.graph.subgraph(com)

                # Extract all edges (relationships) from the community subgraph
                community_triples = []
                for s, t, data in community_subgraph.edges(data=True):
                    triple = (
                        reverse_mapping[s],
                        data.get("label", None),
                        reverse_mapping[t],
                        data.get("source_chunk", None),
                    )
                    community_triples.append(triple)

                communities.append(community_triples)

                # Map original labels to community IDs
                for node in com:
                    node_community_map[reverse_mapping[node]] = com_id
        else:
            # Skip singletons
            pass

    # Print the mapping (optional)
    print("\nNode Community Mapping:")
    for node, community_id in node_community_map.items():
        print(f"Node: {node} -> Community: {community_id}")

    return communities, node_community_map


def summarize_communities(communities, community_map, llm):
    summaries = []
    for community in communities:
        prompt = f"Summarize the following entities and relations from a graph in a paragraph that describes their overall content. Include title and summary.\n {community}"
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], temperature=0.2, seed=42
        )
        output = response["choices"][0]["message"]["content"]
        print(output)
        summaries.append(output)
    return summaries


def embed_communities(summaries):
    """
    Compute embeddings for community summaries using the same model as chunk_document.

    Args:
        summaries: List of summary strings
    Returns:
        List of tuples containing (summary, embedding)
    """
    summaries_with_embeddings = []
    for summary in summaries:
        embedding = model.encode(summary, convert_to_numpy=True)
        summaries_with_embeddings.append((summary, embedding))
    return summaries_with_embeddings


def global_search(summaries, query, llm):
    """
    Perform global search across communities and combine results into a final response.
    """
    intermediate_answers = []
    for index, summary in enumerate(summaries):
        print(f"Processing community {index + 1} of {len(summaries)}")
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"Answer this query based on the summary provided.\nQuery: {query}\nSummary: {summary}",
                }
            ],
            temperature=0.2,
        )
        answer = response["choices"][0]["message"]["content"]
        print(f"Community {index + 1} answer:", answer)
        intermediate_answers.append(answer)

    final_response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"Combine these community answers into a single coherent response:\n{intermediate_answers}",
            }
        ],
        temperature=0.3,
    )

    return final_response["choices"][0]["message"]["content"]


def local_search(query: str, g: nx.Graph, llm):
    """
    Perform local search based on entity-centric reasoning.

    Args:
        query: User query string
        g: NetworkX graph containing entity relationships
    """
    # 1. Extract entities from query
    query_response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"Extract key entities from this query separated by newlines: {query}",
            }
        ],
        temperature=0.1,
    )
    query_entities = query_response["choices"][0]["message"]["content"].split("\n")
    query_entities = [e.strip() for e in query_entities]

    # 2. Find relevant information from graph
    relevant_data = {"entities": set(), "relationships": [], "text_chunks": set()}

    # Find entities and their immediate neighbors
    for query_entity in query_entities:
        for node in g.nodes():
            if query_entity.lower() in str(node).lower():
                # Add entity
                relevant_data["entities"].add(node)

                # Add connected entities and relationships
                for neighbor in g.neighbors(node):
                    relevant_data["entities"].add(neighbor)
                    edge_data = g.get_edge_data(node, neighbor)
                    if edge_data:
                        rel = (node, edge_data.get("label", ""), neighbor)
                        relevant_data["relationships"].append(rel)

                        # Add associated text chunks
                        if "source_chunk" in edge_data:
                            relevant_data["text_chunks"].add(edge_data["source_chunk"])

    # 3. Build context for LLM
    context = f"""
    Relevant Entities: {', '.join(str(e) for e in relevant_data['entities'])}

    Relationships:
    {'\n'.join(f'{s} {r} {t}' for s, r, t in relevant_data['relationships'])}

    Supporting Text:
    {'\n'.join(relevant_data['text_chunks'])}
    """

    # 4. Generate response
    response = llm.create_chat_completion(
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


def main():
    llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_L.gguf",
        local_dir=".",
        verbose=False,
        n_ctx=2048,
        # n_ctx=8192 # Uncomment for larger context windows
    )

    page_chunks = build_chunks("data/Barack_Obama.txt")
    g = build_graph(page_chunks, max_chunks=10)

    nx.write_gpickle(g, "graph.gpickle")

    # Visualize the graph
    visualize(g, "graph.png")

    communities, community_map = build_communities(g)
    # summaries = summarize_communities(communities, community_map, llm)

    # Visualize the communities
    visualize_communities(g, communities, "graph_communities.png")

    # Example global search query
    # query = "What are the main themes in this document?"
    # final_answer = global_search(summaries, query, llm)
    # print("\nFinal Answer:", final_answer)

    # Example local search query
    query = "What did Barack Obama study at Columbia University?"
    local_answer = local_search(query, g, llm)
    print("\nLocal Search Answer:", local_answer)


if __name__ == "__main__":
    main()

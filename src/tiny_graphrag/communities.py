from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
from cdlib import algorithms


@dataclass
class CommunityResult:
    """Represents the result of community detection in a graph.

    Contains the communities found and a mapping of nodes to their community IDs.
    """

    communities: List[List[Tuple[str, str, str, str]]]
    node_community_map: Dict[str, int]


def build_communities(g: nx.Graph) -> CommunityResult:
    """Build communities from a graph."""
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
    # print("\nNode Community Mapping:")
    # for node, community_id in node_community_map.items():
    #     print(f"Node: {node} -> Community: {community_id}")

    return CommunityResult(
        communities=communities, node_community_map=node_community_map
    )

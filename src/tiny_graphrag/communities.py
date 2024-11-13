from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from cdlib import algorithms


@dataclass
class HierarchicalCommunity:
    """Represents a hierarchical community structure."""

    triples: List[Tuple[str, str, str, str]]  # (head, relation, tail, source)
    subcommunities: List["HierarchicalCommunity"]
    summary: Optional[str] = None
    level: int = 0


@dataclass
class CommunityResult:
    """Represents the result of community detection in a graph.

    Contains the hierarchical communities found and a mapping of nodes to their community IDs.
    """

    communities: List[HierarchicalCommunity]
    node_community_map: Dict[str, List[int]]  # Maps nodes to community path


def build_communities(g: nx.Graph) -> CommunityResult:
    """Build communities from a graph."""
    communities: List[HierarchicalCommunity] = []
    node_community_map: Dict[str, List[int]] = {}

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

                # Create HierarchicalCommunity object
                communities.append(
                    HierarchicalCommunity(
                        triples=community_triples, subcommunities=[], level=0
                    )
                )

                # Map original labels to community IDs
                for node in com:
                    node_label = reverse_mapping[node]
                    if node_label not in node_community_map:
                        node_community_map[node_label] = []
                    node_community_map[node_label].append(com_id)
        else:
            # Skip singletons
            pass

    return CommunityResult(
        communities=communities, node_community_map=node_community_map
    )


def build_hierarchical_communities(
    g: nx.Graph, max_levels: int = 3, min_size: int = 5
) -> CommunityResult:
    """Build hierarchical communities from a graph.

    Args:
        g: Input graph
        max_levels: Maximum depth of the hierarchy
        min_size: Minimum size of communities to subdivide
    """
    communities: List[HierarchicalCommunity] = []
    node_community_map: Dict[str, List[int]] = {}

    # Convert to integers while preserving all node attributes
    gi = nx.convert_node_labels_to_integers(g, label_attribute="original_label")

    # Create reverse mapping for nodes
    reverse_mapping = {
        node: data["original_label"] for node, data in gi.nodes(data=True)
    }

    def build_level(subgraph: nx.Graph, level: int) -> List[HierarchicalCommunity]:
        if level >= max_levels or len(subgraph) < min_size:
            return []

        # Detect communities at this level using Leiden algorithm
        coms = algorithms.leiden(subgraph)

        communities = []
        for com_id, com in enumerate(coms.communities):
            # Get the subgraph for this community
            community_subgraph = coms.graph.subgraph(com)

            # Extract triples
            community_triples = []
            for s, t, data in community_subgraph.edges(data=True):
                triple = (
                    reverse_mapping[s],
                    data.get("label", None),
                    reverse_mapping[t],
                    data.get("source_chunk", None),
                )
                community_triples.append(triple)

            # Map nodes to community path
            community_path = [level, com_id]
            for node in com:
                if reverse_mapping[node] not in node_community_map:
                    node_community_map[reverse_mapping[node]] = []
                node_community_map[reverse_mapping[node]].extend(community_path)

            # Recursively build subcommunities
            subcommunities = build_level(community_subgraph, level + 1)

            communities.append(
                HierarchicalCommunity(
                    triples=community_triples,
                    subcommunities=subcommunities,
                    level=level,
                )
            )

        return communities

    # Start building hierarchy from connected components
    for component in nx.connected_components(gi):
        subgraph = gi.subgraph(component)
        if len(subgraph) > 1:
            communities.extend(build_level(subgraph, 0))

    return CommunityResult(
        communities=communities, node_community_map=node_community_map
    )

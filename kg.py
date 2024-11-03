import networkx as nx
from typing import Dict, List, Set, Optional, Union
from dataclasses import dataclass
from tinyrog.extract import extract_ents, extract_rels
import leidenalg
import igraph as ig
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


@dataclass
class Entity:
    text: str
    type: str


@dataclass
class Relationship:
    head: str
    relation: str
    tail: str


@dataclass
class KnowledgeGraph:
    entities: Dict[str, Entity]
    relationships: List[Relationship]
    graph: nx.Graph
    communities: Dict[str, Set[str]]

    @classmethod
    def from_text(cls, text: str, resolution: float = 1.0) -> "KnowledgeGraph":
        # Extract entities and relationships
        entities = extract_ents(text)
        relationships = extract_rels(text)

        # Create networkx graph
        G = nx.Graph()

        # Add entities as nodes
        entity_dict = {}
        for ent_text, ent_type in entities:
            entity_dict[ent_text] = Entity(text=ent_text, type=ent_type)
            G.add_node(ent_text, type=ent_type)

        # Add relationships as edges and ensure all entities exist
        rel_list = []
        for head, rel, tail in relationships:
            # Add any missing entities from relationships
            if head not in entity_dict:
                # entity_dict[head] = Entity(text=head, type="UNKNOWN")
                # G.add_node(head, type="UNKNOWN")
                print(
                    f"Skipping relationship {head} {rel} {tail} because {head} not in entities"
                )
            if tail not in entity_dict:
                # entity_dict[tail] = Entity(text=tail, type="UNKNOWN")
                # G.add_node(tail, type="UNKNOWN")
                print(
                    f"Skipping relationship {head} {rel} {tail} because {tail} not in entities"
                )

            G.add_edge(head, tail, relation=rel)
            rel_list.append(Relationship(head=head, relation=rel, tail=tail))

        # Convert networkx graph to igraph for Leiden algorithm
        ig_graph = ig.Graph.from_networkx(G)

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )

        # Group entities by community
        community_groups: Dict[str, Set[str]] = defaultdict(set)
        for idx, community in enumerate(partition):
            for vertex_idx in community:
                node = ig_graph.vs[vertex_idx]["_nx_name"]  # Get original node name
                community_groups[str(idx)].add(node)

        return cls(
            entities=entity_dict,
            relationships=rel_list,
            graph=G,
            communities=community_groups,
        )

    def get_community_summary(self, community_id: str) -> str:
        """Generate a summary for a specific community"""
        entities = self.communities[community_id]

        # Get all relationships within this community
        community_rels = []
        for rel in self.relationships:
            if rel.head in entities and rel.tail in entities:
                community_rels.append(rel)

        # Build summary
        summary = []
        summary.append(f"Community {community_id} contains {len(entities)} entities:")

        # List entities by type
        type_groups = defaultdict(list)
        for ent in entities:
            entity = self.entities[ent]
            type_groups[entity.type].append(entity.text)

        for ent_type, ents in type_groups.items():
            summary.append(f"\n{ent_type}: {', '.join(ents)}")

        if community_rels:
            summary.append("\nRelationships:")
            for rel in community_rels:
                summary.append(f"- {rel.head} {rel.relation} {rel.tail}")

        return "\n".join(summary)

    def optimize_communities(self, resolution: Optional[float] = None) -> None:
        """Rerun community detection with optional new resolution parameter"""
        if resolution is None:
            resolution = 1.0

        # Convert networkx graph to igraph
        ig_graph = ig.Graph.from_networkx(self.graph)

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )

        # Update communities
        self.communities.clear()
        for idx, community in enumerate(partition):
            for vertex_idx in community:
                node = ig_graph.vs[vertex_idx]["_nx_name"]
                self.communities[str(idx)].add(node)

    def visualize(
        self,
        output_path: Union[str, Path] = "knowledge_graph.png",
        figsize: tuple[int, int] = (20, 20),
    ) -> None:
        """
        Visualize the knowledge graph with communities in different colors.
        Nodes are sized by their degree centrality.
        """
        # Create figure
        plt.figure(figsize=figsize)

        # Get node positions using spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)

        # Calculate node sizes based on degree centrality
        centrality = nx.degree_centrality(self.graph)
        [v * 3000 for v in centrality.values()]

        # Generate colors for communities
        colors = list(mcolors.TABLEAU_COLORS.values())
        # if len(self.communities) > len(colors):
        #     colors = plt.cm.tab20(np.linspace(0, 1, len(self.communities)))

        # Draw nodes for each community
        for idx, (community_id, nodes) in enumerate(self.communities.items()):
            color = colors[idx % len(colors)]
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=list(nodes),
                node_size=[centrality[node] * 3000 for node in nodes],
                node_color=[color] * len(nodes),
                alpha=0.6,
                label=f"Community {community_id}",
            )

        # Draw edges
        edge_labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, edge_color="gray")

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight="bold")

        # Add edge labels
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=6
        )

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Remove axes
        plt.axis("off")

        # Save to file
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(str(output_path), bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Graph visualization saved to {output_path}")

    def visualize_communities(
        self, output_dir: Union[str, Path] = "community_graphs"
    ) -> None:
        """
        Generate separate visualizations for each community.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for community_id, nodes in self.communities.items():
            # Create subgraph for this community
            subgraph = self.graph.subgraph(nodes)

            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(subgraph, k=1, iterations=50)

            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph, pos, node_size=1000, node_color="lightblue", alpha=0.6
            )

            # Draw edges
            edge_labels = nx.get_edge_attributes(subgraph, "relation")
            nx.draw_networkx_edges(subgraph, pos, alpha=0.4)

            # Draw labels
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight="bold")

            # Add edge labels
            nx.draw_networkx_edge_labels(
                subgraph, pos, edge_labels=edge_labels, font_size=6
            )

            plt.title(f"Community {community_id}")
            plt.axis("off")

            # Save to file
            community_file = output_path / f"community_{community_id}.png"
            plt.savefig(str(community_file), bbox_inches="tight", dpi=300)
            plt.close()

        print(f"Community visualizations saved to {output_path}/")


def build_knowledge_graph(text: str) -> KnowledgeGraph:
    """Build a knowledge graph from input text"""
    return KnowledgeGraph.from_text(text)


def main() -> None:
    text = """
    Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Hawthorne, California.
    The company develops rockets for space travel. Tesla, another company led by Musk,
    is based in Austin, Texas. In 2022, Musk acquired Twitter, which was founded by
    Jack Dorsey in 2006. Twitter is now known as X and is based in San Francisco.
    
    Mark Zuckerberg created Facebook in 2004 while at Harvard University. Facebook
    later became Meta Platforms and acquired Instagram and WhatsApp. Meta is
    headquartered in Menlo Park, California.
    """

    print("Building knowledge graph...")
    # Create knowledge graph with default resolution
    kg = build_knowledge_graph(text)

    # Print basic statistics
    print(
        f"\nFound {len(kg.entities)} entities and {len(kg.relationships)} relationships"
    )

    # Print all entities by type
    print("\nEntities by type:")
    type_groups = defaultdict(list)
    for entity in kg.entities.values():
        type_groups[entity.type].append(entity.text)

    for ent_type, entities in type_groups.items():
        print(f"{ent_type}: {', '.join(entities)}")

    # Print all relationships
    print("\nRelationships:")
    for rel in kg.relationships:
        print(f"{rel.head} {rel.relation} {rel.tail}")

    # Print initial community detection results
    print("\nInitial Communities (resolution=1.0):")
    for community_id in kg.communities:
        print("\n" + kg.get_community_summary(community_id))

    # Add visualization after community detection
    print("\nGenerating visualizations...")
    kg.visualize("outputs/full_graph.png")
    kg.visualize_communities("outputs/communities")

    # Try different resolution for community detection
    print("\nOptimizing communities with higher resolution (1.5)...")
    kg.optimize_communities(resolution=1.5)

    # Visualize the optimized communities
    print("\nGenerating visualizations for optimized communities...")
    kg.visualize("outputs/full_graph_optimized.png")
    kg.visualize_communities("outputs/communities_optimized")

    print("\nUpdated Communities:")
    for community_id in kg.communities:
        print("\n" + kg.get_community_summary(community_id))


if __name__ == "__main__":
    main()

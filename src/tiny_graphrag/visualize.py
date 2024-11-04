from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors as mcolors


def visualize(
    graph: nx.Graph,
    output_path: str = "graph.png",
    figsize: tuple[int, int] = (10, 10),
    node_size: int = 500,
    node_color: str = "#1f78b4",
    font_size: int = 10,
) -> None:
    """Visualize a networkx graph and save it as a PNG file.

    Args:
        graph: A networkx graph object to visualize
        output_path: Path where the PNG file should be saved
        figsize: Tuple of (width, height) for the figure size
        node_size: Size of the nodes in the visualization
        node_color: Color of the nodes
        font_size: Size of the node labels
    """
    # Create a new figure with specified size
    plt.figure(figsize=figsize)

    # Create the layout for the graph
    pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        node_size=node_size,
        with_labels=True,
        font_size=font_size,
        font_weight="bold",
    )

    # Save the graph to a file
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()


def visualize_communities(
    graph: nx.Graph,
    communities: List[List[Tuple[str, str, str, str]]],
    output_path: Union[str, Path] = "graph_communities.png",
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """Visualize a networkx graph with communities in different colors.

    Args:
        graph: A networkx graph object to visualize
        communities: Dictionary mapping community IDs to sets of node names
        output_path: Path where the PNG file should be saved
        figsize: Tuple of (width, height) for the figure size
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Get node positions using spring layout
    pos = nx.spring_layout(graph, k=1, iterations=50)

    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(graph)
    node_sizes = {node: cent * 3000 for node, cent in centrality.items()}

    # Generate colors for communities
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Draw nodes for each community
    for idx, nodes in enumerate(communities):
        color = colors[idx % len(colors)]
        # Turn the triples into a list of nodes
        node_list = list(
            chain.from_iterable([[triple[0], triple[2]] for triple in nodes])
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=node_list,
            node_size=[node_sizes[node] for node in node_list],
            node_color=[color] * len(node_list),
            alpha=0.6,
            label=f"Community {idx}",
        )

    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color="gray")

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

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

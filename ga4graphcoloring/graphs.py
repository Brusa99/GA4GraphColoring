import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings

# suppress warnings from networkx about no data for colormapping provided, as we want the default color in this case
warnings.filterwarnings("ignore", message="No data for colormapping provided")


class Graph:
    """(Undirected) Graph class"""

    def __init__(self, n_vertices: int, density_factor: float = 0.5):
        """Constructor method.

        Randomly generates a symmetric adjacency matrix for the graph, probability of an edge existing between two
         vertices is given by the density_factor.
        """
        self.n_vertices = n_vertices
        self.density_factor = density_factor

        # Randomly generate adjacency matrix, edge exists with probability `density_factor`
        adj_matrix = np.random.rand(n_vertices, n_vertices)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix < density_factor).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        self.adj_matrix = adj_matrix

    def __repr__(self):
        return f"Graph (n_vertices={self.n_vertices}, density_factor={self.density_factor})"

    def display(self, colors: list[int] | None = None):
        """Display graph using networkx"""

        if not colors:  # if no color is provided use default color
            rescaled_colors = "#1f78b4"
        else:
            assert len(colors) == self.n_vertices, "Coloring must match number of vertices"
            # color arg must be in range [0, 1]
            max_color = max(colors)
            rescaled_colors = colors / max_color

        # draw graph using networkx
        G = nx.from_numpy_array(self.adj_matrix)
        nx.draw(G, with_labels=True, font_weight='bold', node_color=rescaled_colors, cmap="gnuplot")

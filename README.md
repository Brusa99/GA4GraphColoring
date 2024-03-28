# GA4GraphColoring

Project for the 2023-24 course of Global and Multi-Objective Optimization at Units.
The goal of the project is to implement a Genetic Algorithm to solve the _Graph Coloring problem_.

## Description

In this project we implement two Genetic Algorithms to solve the Graph Coloring problem.

The Graph Coloring problem consists in assigning a color to each vertex of a graph such that no two adjacent vertices
have the same color. It is a NP-hard problem.

In particular, we tackle the decision version of the problem, that is, given a graph $G$ and an integer $k$, we want to
know if it is possible to color the graph with $k$ colors. The decision version of the problem is a NP-complete problem.

The genetic algorithm is implemented in two ways: a _naive_ version and a more problem-specific version.
the _naive_ version is a general implementation of a genetic algorithm, while the problem-specific uses more specific
selection and mutation operators.

Additionally, sudoku problems are a special case of the Graph Coloring problem, where the graph is fixed and a
9-coloring is searched. An implementation of the algorithm described above is used to solve sudoku problems.

## Installation

To install the project, clone the repository and install the requirements:

```bash
git clone git@github.com:Brusa99/GA4GraphColoring.git
cd GA4GraphColoring
pip install -r requirements.txt
```

Additionally, to run the [examples](examples) `jupyter` and `matplotlib` are required.

## Usage

The project can be used as a library, for example:

```python
import ga4graphcoloring as ga

# Create a graph and a population to color it
graph = ga.Graph(n_vertices=10, density_factor=0.5)
pop = ga.Population(max_colors=4, pop_size=20, graph=graph)

# Run the genetic algorithm
for it in range(100):
    pop.evolve()
    if pop.best_fitness == 0:
        print(f"Solution found in {it} iterations")
        print(f"Solution: {pop.solution}")
        break
```

For more examples, see the [examples](examples) directory.


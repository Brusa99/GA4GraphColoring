from typing import Optional
import random

import numpy as np

from ga4graphcoloring.gen_algs import Population, SmartPopulation
from ga4graphcoloring.ga4sudoku.board import Sudoku


class SudokuPopulation(Population):
    """Population of individuals for sudoku solving.

    This class represents a population of individuals for solving the sudoku problem.
    The algorithm is a particular case of the genetic algorithm for graph coloring, where the graph is the sudoku board.
    Selection, crossover and fitness are performed as in the base population genetic algorithm, but the mutation
    function is modified to respect the given sudoku board.

    Attributes:
        pop_size (int): Size of the population.
        sudoku (Sudoku): Sudoku object to solve.
        blocked_cells (np.ndarray): Indices of given cells in the sudoku board, mutation does not affect these cells.
        genotype_size (int): Number of cells in the sudoku board, fixed to 81.
    """

    def __init__(self, pop_size: int, sudoku: Sudoku):
        """Randomly initialize a population of individuals for the sudoku problem.

        Blocked cells are gathered by the sudoku object and are initialized with their value.
        The rest of the cells are initialized with a random number between 1 and 9 (included).

        Args:
            pop_size: Size of the population.
            sudoku: Sudoku object to solve.
        """
        super().__init__(9, pop_size, sudoku)
        # self.pop_size = pop_size
        self.sudoku = self.graph

        self._genotype_range = list(range(1, 10))
        self.genotype_size = 81

        # get blocked cells
        self.blocked_cells = np.where(self.sudoku.value_matrix.flatten() != 0)[0]

        # initialize the population with random individuals of size 81
        self.individuals = [self._initialize_individual() for _ in range(self.pop_size)]
        self.individuals.sort(key=self.fitness)

    def _initialize_individual(self) -> np.ndarray:
        """Initialize a random individual for the sudoku problem."""

        individual = np.zeros(self.genotype_size, dtype=int)
        # blocked cells are initialized with their value
        for blocked in self.blocked_cells:
            individual[blocked] = self.sudoku.value_matrix.flatten()[blocked]
        # free cells are initialized with a random number
        for i in range(self.genotype_size):
            if individual[i] == 0:
                individual[i] = np.random.choice(self._genotype_range)
        return individual

    def mutate(self, individual: np.ndarray, mutation_rate: Optional[float] = None) -> np.ndarray:
        """Mutate an individual.

        Mutation is performed by changing the value of a random cell to a random number between 1 and 9 (included).
        Blocked cells are not mutated.
        The probability of mutation is given by the mutation_rate. It defaults to 1/n_free_cells.

        Args:
            individual: The individual to mutate.
            mutation_rate: The probability of mutation. Defaults to 1/n_free_cells.

        Returns:
            np.ndarray: The mutated individual.
        """

        # set mutation rate to 1/n_free_cells if not provided
        if mutation_rate is None:
            mutation_rate = 1 / (self.genotype_size - len(self.blocked_cells))

        for i in range(self.genotype_size):
            if np.random.rand() < mutation_rate and i not in self.blocked_cells:
                individual[i] = np.random.choice(self._genotype_range)
        return individual


class SmartSudokuPopulation(SmartPopulation, SudokuPopulation):
    """Implementation of SmartPopulation for the sudoku problem.

    This class represents a population of individuals for solving the sudoku problem using the SmartPopulation strategy.
    The class adapts the SmartPopulation methods to fit the sudoku problem. In particular: mutation methods don't change
    the value of blocked cells.
    """

    def __init__(self, pop_size: int, sudoku: Sudoku, change_operator_threshold: int = 4):
        """Randomly initialize a population of individuals for the sudoku problem.

        Blocked cells are gathered by the sudoku object and are initialized with their value.
        The rest of the cells are initialized with a random number between 1 and 9 (included).

        Args:
            pop_size: Size of the population.
            sudoku: Sudoku object to solve.
            change_operator_threshold (int): The fitness threshold for changing the operators. Defaults to 4.
        """
        self._genotype_range = list(range(1, 10))
        SudokuPopulation.__init__(self, pop_size, sudoku)
        self.change_operator_threshold = change_operator_threshold

    def _adj_mutation(self, individual: np.ndarray, mutation_rate: float = 0.7) -> np.ndarray:
        """Mutation operator 1: Adjacency mutation. Change color of violating vertex with a not adjacent color."""

        for vertex_ind, color in enumerate(individual):
            if np.random.rand() < mutation_rate and vertex_ind not in self.blocked_cells:
                # get adjacent vertices colors
                adj_colors = individual[self.graph.adj_matrix[vertex_ind].nonzero()[0]]  # individual[adjacent_vertices]

                # if the vertex has the same color as any of its neighbours, change its color with a not adjacent color
                if color in adj_colors:
                    valid_colors = set(range(self.max_colors)) - set(adj_colors) - {0}
                    if valid_colors:  # if there are valid colors
                        individual[vertex_ind] = random.choice(list(valid_colors))
                    # TODO: consider selecting a random color if there are none available (paper is not clear on this)
        return individual

    def _random_mutation(self, individual: np.ndarray, mutation_rate: float = 0.7) -> np.ndarray:
        """Mutation operator 2: Random mutation. Change color of violating vertex with a random color."""

        for vertex_ind, color in enumerate(individual):
            if np.random.rand() < mutation_rate and vertex_ind not in self.blocked_cells:
                # get adjacent vertices colors
                adj_colors = individual[self.graph.adj_matrix[vertex_ind].nonzero()[0]]  # individual[adjacent_vertices]

                # if the vertex has the same color as any of its neighbours, change its color with a random color
                if color in adj_colors:
                    individual[vertex_ind] = random.choice(self._genotype_range)
        return individual

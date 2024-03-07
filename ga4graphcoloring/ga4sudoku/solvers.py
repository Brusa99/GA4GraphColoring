from typing import Optional

import numpy as np

from ga4graphcoloring.gen_algs import Population
from ga4graphcoloring.ga4sudoku.board import Sudoku


class SudokuPopulation(Population):
    """Population of individuals for sudoku solving.

    This class represents a population of individuals for the sudoku problem.
    The algorithm is a particular case of the genetic algorithm for graph coloring, where the graph is the sudoku board.
    Selection, crossover and fitness are performed as in the general genetic algorithm, but the mutation function is
    modified to respect the given sudoku board.
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




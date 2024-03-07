from typing import Optional

import numpy as np

from ga4graphcoloring.gen_algs import Population
from ga4graphcoloring.ga4sudoku.board import Sudoku


class SudokuPopulation(Population):

    def __init__(self, pop_size: int, sudoku: Sudoku):

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
        """Mutate an individual."""

        # set mutation rate to 1/n_free_cells if not provided
        if mutation_rate is None:
            mutation_rate = 1 / (self.genotype_size - len(self.blocked_cells))

        for i in range(self.genotype_size):
            if np.random.rand() < mutation_rate and i not in self.blocked_cells:
                individual[i] = np.random.choice(self._genotype_range)
        return individual




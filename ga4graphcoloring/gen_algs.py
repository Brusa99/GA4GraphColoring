import random

import numpy as np

from ga4graphcoloring.graphs import Graph


class Population:
    """Population of individuals for graph coloring.

    This class represents a population of individuals for the graph coloring problem.
    The population is initialized with random individuals, and can evolve through selection, crossover and mutation.
    The indivials are represented as numpy arrays of size n_vertices, where each element represents the color of the
    corresponding vertex. Each individual can have a maximum of `max_colors` colors.

    Attributes:
        max_colors (int): The maximum number of colors an individual can have.
        pop_size (int): The size of the population.
        graph (Graph): The graph to color.
        genotype_size (int): The size of the genotype of the individuals.
        individuals (list[np.ndarray]): The list of individuals in the population.

    Methods:
        fitness: Calculate the fitness of an individual.
        crossover: Perform one point crossover between two parents to produce a child.
        mutate: Mutate an individual.
        selection: Select an individual from the population using tournament selection.
        evolve: Perform one generation of evolution.
        solution: Return the best individual in the population.
    """

    def __init__(self, max_colors: int, pop_size: int, graph: Graph):
        self.max_colors = max_colors
        self.pop_size = pop_size
        self.graph = graph

        # initialize the population with random individuals of size n_vertices
        self.genotype_size = self.graph.n_vertices
        self.individuals = [np.random.randint(max_colors, size=self.genotype_size) for _ in range(self.pop_size)]

    def fitness(self, individual: np.ndarray) -> int:
        """Calculate the fitness for an individual.

        Fitness is calculated as the number of edges  the same color.

        Args:
            individual: The individual to calculate the fitness for.
        Returns:
            float: The fitness value for the individual.
        """
        fitness = 0

        # color of a vertex is given by the value stored the corresponding index of the individual
        for color, adj_row in zip(individual, self.graph.adj_matrix):
            # check if the vertex has the same color as any of its neighbours
            for vertex in adj_row.nonzero()[0]:  # only iter on adjacent vertices
                if color == individual[vertex]:
                    fitness += 1
        # fitness is halved as we count each edge twice
        return fitness // 2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Performs one point crossover between two parents.

        This method performs a one point crossover between two parents to produce a (single) child.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.
        Returns:
            np.ndarray: The child individual produced by the crossover.
        """
        crossover_point = np.random.randint(1, self.genotype_size)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutate(self, individual: np.ndarray, mutation_rate: float | None = None) -> np.ndarray:
        """Mutate an individual.

        This method performs a mutation on an individual with a given mutation rate, defaulting to 1/n_vertices.
        The mutation consists of randomly changing the color of a vertex.

        Args:
            individual (np.ndarray): The individual to mutate.
            mutation_rate (int): The mutation rate.
        Returns:
            np.ndarray: The mutated individual.
        """
        # default mutation rate is 1/n_vertices
        if mutation_rate is None:
            mutation_rate = 1 / self.genotype_size

        offspring = individual.copy()
        for i in range(self.genotype_size):
            if np.random.rand() < mutation_rate:
                offspring[i] = np.random.randint(self.max_colors)
        return offspring

    def selection(self, tournament_size: int | None = None) -> np.ndarray:
        """Select an individual from the population using tournament selection.

        This method selects an individual from the population using tournament selection.
        Selection pressure is controlled by the tournament size, which defaults to 5.

        Args:
            tournament_size (int): The size of the tournament. Defaults to 5.
        Returns:
            np.ndarray: The selected individual.
        """
        if tournament_size is None:
            tournament_size = 5

        # tournament = np.random.choice(self.individuals, size=tournament_size, replace=False)
        tournament = random.choices(self.individuals, k=tournament_size)
        fitness_scores = [self.fitness(individual) for individual in tournament]
        return tournament[np.argmin(fitness_scores)]

    def evolve(self,
               mutation_rate: float | None = None,
               tournament_size: int | None = None,
               elitism: bool = True) -> None:
        """Perform one generation of evolution.

        This method performs one generation of evolution on the population, consisting of:
        - Crossover: Pairs of parents are selected from the current population and crossed over to produce children. The
        selection of parents is done using tournament selection.
        - Mutation: The children are mutated with a given mutation rate.

        Args:
            mutation_rate (float): The mutation rate. Defaults to 1/n_vertices.
            tournament_size (int): The size of the tournament. Defaults to 5.
            elitism (bool): Whether to use elitism. Defaults to True.
        """
        new_population = []
        if elitism:
            n_elites = int(0.1 * self.pop_size)  # round to nearest integer
            # carry the best (first of the list) to the new population
            elites = sorted(self.individuals, key=self.fitness)[:n_elites]
            new_population.extend(elites)

        while len(new_population) < self.pop_size:
            # select parents with tournament selection
            parent1 = self.selection(tournament_size)
            parent2 = self.selection(tournament_size)

            # crossover the parents to produce a child and mutate it
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate)

            # add the child to the new population
            new_population.append(child)

        # replace the population
        self.individuals = new_population

    @property
    def solution(self) -> np.ndarray:
        """Best individual in the population.

        Returns:
            np.ndarray: The best individual in the population.
        """
        return min(self.individuals, key=self.fitness)

    @property
    def best_fitness(self) -> int:
        """Best fitness value in the population.

        Returns:
            int: The fitness value of the best individual in the population.
        """
        return self.fitness(self.solution)

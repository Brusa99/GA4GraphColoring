import random
from typing import Optional

import numpy as np

from ga4graphcoloring.graphs import Graph


class Population:
    """Population of individuals for graph coloring.

    This class represents a population of individuals for the graph coloring problem.
    The population is initialized with random individuals, and can evolve through selection, crossover and mutation.
    The indivials are represented as numpy arrays of size `n_vertices`, where each element represents the color of the
    corresponding vertex. Each individual can have a maximum of `max_colors` colors. Indviduals are kept sorted by
    fitness to speed up best fitness check, elitism and derived class evolution.

    Attributes:
        max_colors (int): The maximum number of colors an individual can have.
        pop_size (int): The size of the population.
        graph (Graph): The graph to color.
        genotype_size (int): The size of the genotype of the individuals.
        individuals (list[np.ndarray]): The list of individuals in the population, sorted by fitness.
    """

    def __init__(self, max_colors: int, pop_size: int, graph: Graph):
        """Constructor method.

        Initializes the population with random individuals of size graph.n_vertices. Individuals are sorted by fitness.

        Args:
            max_colors (int): The maximum number of colors an individual can have.
            pop_size (int): The size of the population.
            graph (Graph): The target graph to color.
        """
        self.max_colors = max_colors
        self.pop_size = pop_size
        self.graph = graph

        # initialize the population with random individuals of size n_vertices
        self.genotype_size = self.graph.n_vertices
        self.individuals = [np.random.randint(max_colors, size=self.genotype_size) for _ in range(self.pop_size)]

        # sort the individuals by fitness to speed up best fitness check, elitism and derived class evolution
        self.individuals.sort(key=self.fitness)

    def fitness(self, individual: np.ndarray) -> int:
        """Calculate the fitness for an individual.

        Fitness is calculated as the number of edges connected to same color vertices.

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
        The crossover point is randomly selected between 0 and `genotype_size` (excluded).

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.
        Returns:
            np.ndarray: The child individual produced by the crossover.
        """
        crossover_point = np.random.randint(0, self.genotype_size)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutate(self, individual: np.ndarray, mutation_rate: Optional[float] = None) -> np.ndarray:
        """Mutate an individual.

        This method performs a mutation on an individual with a given mutation rate, defaulting to 1/`n_vertices`.
        The mutation consists of assigning a random color to a vertex, for each vertex of the individual with the given
        probability.

        Args:
            individual (np.ndarray): The individual to mutate.
            mutation_rate (float): The mutation rate. Defaults to 1/n_vertices.
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

    def selection(self, tournament_size: Optional[int] = None) -> tuple[np.ndarray]:
        """Select two individuala from the population using tournament selection.

        This method selects an individual from the population using tournament selection.
        The process is repeated twice to select two individuals.
        Selection pressure is controlled by the tournament size.

        Args:
            tournament_size (int): The size of the tournament. Defaults to 5.
        Returns:
            np.ndarray: The selected individual.
        """
        if tournament_size is None:
            tournament_size = 5

        selected = []
        for _ in range(2):
            tournament = random.choices(self.individuals, k=tournament_size)
            fitness_scores = [self.fitness(individual) for individual in tournament]
            selected.append(tournament[np.argmin(fitness_scores)])
        return selected[0], selected[1]

    def evolve(self,
               mutation_rate: Optional[float] = None,
               tournament_size: Optional[int] = None,
               elitism: bool = True) -> None:
        """Perform one generation of evolution.

        This method performs one generation of evolution on the population, consisting of:
        - Crossover: Pairs of parents are selected from the current population and crossed over to produce children. The
        selection of parents is done using tournament selection.
        - Mutation: The children are mutated with a given mutation rate.

        The population is replaced with the children, keeping the best individuals if elitism is enabled.
        The method also sorts the population by fitness after the evolution.

        Args:
            mutation_rate (float): The mutation rate. Defaults to 1/n_vertices.
            tournament_size (int): The size of the tournament. Defaults to 5.
            elitism (bool): Whether to use elitism. Defaults to True.
        """
        new_population = []
        if elitism:
            n_elites = int(0.1 * self.pop_size)  # round to nearest integer
            # carry the best (first of the list) to the new population
            elites = self.individuals[:n_elites]
            new_population.extend(elites)

        while len(new_population) < self.pop_size:
            # select parents with tournament selection
            parent1, parent2 = self.selection(tournament_size)

            # crossover the parents to produce a child and mutate it
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate)

            # add the child to the new population
            new_population.append(child)

        # replace the population
        new_population.sort(key=self.fitness)
        self.individuals = new_population

    @property
    def solution(self) -> np.ndarray:
        """Best individual in the population."""
        return self.individuals[0]  # individuals are sorted by fitness

    @property
    def best_fitness(self) -> int:
        """Best fitness value in the population."""
        return self.fitness(self.solution)


class SmartPopulation(Population):
    """Population of individuals with more suited operators for graph coloring.

    This class is an improvement on the base Population class, with more suited operators for graph coloring problem.
    The operators are taken from [1]. The main differences are:
        - Two types of selection: tournament selection with size 2 tournament and top genome selection
                                  (the best two individuals are selected as parents).
        - Two types of mutation: random mutation and adjacency mutation. Only same color adjacent vertices are mutated.
        - At each evolution step, half of the population (with worse fitness) is replaced with random individuals.
    The class decides which operator to use based on the population fitness and a threshold.

    Attributes:
        max_colors (int): The maximum number of colors an individual can have.
        pop_size (int): The size of the population.
        graph (Graph): The graph to color.
        genotype_size (int): The size of the genotype of the individuals.
        individuals (list[np.ndarray]): The list of individuals in the population, sorted by fitness.
        change_operator_threshold (int): The fitness threshold for changing the operators. Defaults to 4.

    References:
        [1] Hindi, Musa & Yampolskiy, Roman. (2012). Genetic Algorithm Applied to the Graph Coloring Problem.
            Midwest Artificial Intelligence and Cognitive Science Conference. 60.
    """
    def __init__(self, max_colors: int, pop_size: int, graph: Graph, change_operator_threshold: int = 4):
        """Constructor method, extends base class.

        Initializes the population with random individuals of size graph.n_vertices. Individuals are sorted by fitness.

        Args:
            max_colors (int): The maximum number of colors an individual can have.
            pop_size (int): The size of the population.
            graph (Graph): The target graph to color.
            change_operator_threshold (int): The fitness threshold for changing the operators. Defaults to 4.
        """

        super().__init__(max_colors, pop_size, graph)
        self.change_operator_threshold = change_operator_threshold

    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.7) -> np.ndarray:
        """Mutate an individual.

        This method performs a mutation on an individual.
        The mutation operator is chosen based on the population fitness:
        - If the best fitness is greater than `change_operator_threshold`, the adjacency mutation operator is used.
        - Otherwise, the random mutation operator is used.

        The adjacency mutation operator changes the color of a violating vertex with a not adjacent color.
        The random mutation operator changes the color of a violating vertex with a random color.

        The method overrides the base class mutate method.

        Args:
            individual (np.ndarray): The individual to mutate.
            mutation_rate (float): The mutation rate. Defaults to 0.7.

        Returns:
            np.ndarray: The mutated individual.
        """
        if self.best_fitness > self.change_operator_threshold:
            return self._adj_mutation(individual, mutation_rate)
        else:
            return self._random_mutation(individual, mutation_rate)

    def _adj_mutation(self, individual: np.ndarray, mutation_rate: float = 0.7) -> np.ndarray:
        """Mutation operator 1: Adjacency mutation. Change color of violating vertex with a not adjacent color."""

        for vertex_ind, color in enumerate(individual):
            if np.random.rand() < mutation_rate:
                # get adjacent vertices colors
                adj_colors = individual[self.graph.adj_matrix[vertex_ind].nonzero()[0]]  # individual[adjacent_vertices]

                # if the vertex has the same color as any of its neighbours, change its color with a not adjacent color
                if color in adj_colors:
                    valid_colors = set(range(self.max_colors)) - set(adj_colors)
                    if valid_colors:  # if there are valid colors
                        individual[vertex_ind] = random.choice(list(valid_colors))
                    # TODO: consider selecting a random color if there are none available (paper is not clear on this)
        return individual

    def _random_mutation(self, individual: np.ndarray, mutation_rate: float = 0.7) -> np.ndarray:
        """Mutation operator 2: Random mutation. Change color of violating vertex with a random color."""

        for vertex_ind, color in enumerate(individual):
            if np.random.rand() < mutation_rate:
                # get adjacent vertices colors
                adj_colors = individual[self.graph.adj_matrix[vertex_ind].nonzero()[0]]  # individual[adjacent_vertices]

                # if the vertex has the same color as any of its neighbours, change its color with a random color
                if color in adj_colors:
                    individual[vertex_ind] = np.random.randint(self.max_colors)
        return individual

    def selection(self, tournament_size: Optional[int] = 2) -> tuple[np.ndarray]:
        """Select two individuals from the population.

        This method selects an individual from the population using two different selection operators, based on the
        population fitness:
        - If the best fitness is greater than `change_operator_threshold`, the tournament selection operator is used.
        - Otherwise, the top genome selection operator is used.

        The process is repeated twice to select two individuals.

        The method overrides the base class selection method.

        Args:
            tournament_size: Size of the tournament. Only for tournament selection. Defaults to 2.

        Returns:
            np.ndarray: The selected individual.
        """
        if self.best_fitness > self.change_operator_threshold:
            return self._tournament_selection(tournament_size)
        else:
            return self._top_genome_selection()

    def _tournament_selection(self, tournament_size: int = 2) -> tuple[np.ndarray]:
        """Selection operator 1: Tournament selection."""

        return super().selection(tournament_size)

    def _top_genome_selection(self) -> tuple[np.ndarray]:
        """Selection operator 2: Top genome selection."""

        return self.individuals[0], self.individuals[1]  # individuals are sorted by fitness

    def evolve(self, mutation_rate: float = 0.7, tournament_size: int = 2, **kwargs):
        """Perform one generation of evolution.

        This method performs one generation of evolution on the population.
        Before evolving, the worst half of the population is replaced with random individuals.
        The evolution consists of selection, crossover and mutation.

        The method extends the base class evolve method by replacing the worst half of the population with random
        individuals before performing the evolution.

        Args:
            mutation_rate (float): The mutation rate. Defaults to 0.7.
            tournament_size (int): The size of the selection tournament. Defaults to 2.
        """
        # replace the worst half of the population with random individuals
        self.individuals = self.individuals[:self.pop_size // 2]  # individuals are sorted by fitness
        while len(self.individuals) < self.pop_size:
            self.individuals.append(np.random.randint(self.max_colors, size=self.genotype_size))

        # perform evolution as in base class
        super().evolve(mutation_rate, tournament_size, elitism=False)

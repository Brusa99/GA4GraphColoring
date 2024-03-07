from typing import Optional

import numpy as np
from ga4graphcoloring.graphs import Graph
import sudoku as pysudoku


class SudokuTemplate(Graph):
    """Template for a sudoku board to be filled in by the genetic algorithm.

    This class represents a sudoku board as a graph, where each node represents a cell in the board.
    The graph is generated with 81 vertices and fixed edjes according to the rules of sudoku. That means that each node
    is connected to all other nodes in the same row, column and block.
    Node are numbered from 0 to 80, starting from the top left corner and going row by row.

    Attributes:
        adj_matrix (np.ndarray): The adjacency matrix of the graph.
        value_matrix (np.ndarray): The value matrix of the sudoku board. "0" represents an empty cell.

    Methods:
        display: Display the sudoku board.

    Notes:
        The sudoku is blanck, to generate a partially full sudoku (to be solved) see the `Sudoku` class.
    """

    def __init__(self):
        super().__init__(81, 0)

        # init the adjacency matrix
        adj_matrix = np.zeros((81, 81))
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    # row: a node is connected to all other nodes in the same row
                    adj_matrix[i * 9 + j, i * 9 + k] = 1
                    # column: a node is connected to all other nodes in the same column
                    adj_matrix[i * 9 + j, k * 9 + j] = 1
                    # block: a node is connected to all other nodes in the same block
                    adj_matrix[i * 9 + j, (i // 3 * 3 + k // 3) * 9 + j // 3 * 3 + k % 3] = 1
        # remove self loops
        np.fill_diagonal(adj_matrix, 0)
        self.adj_matrix = adj_matrix

        self.value_matrix = np.zeros((9, 9), dtype=int)

    def display(self, colors: Optional[list[int]] = None):
        """Display the sudoku board.

        Prints the sudoku board to the console. If `colors` are provided, the board is printed with the requested
        numbers. `colors` should be a list of 81 integers, where each integer represents the value of the corresponding
        cell in the sudoku board. Enumberation is done row by row, starting from the top left corner.

        Args:
            colors (list[int]): Vector of length 81 with the numbers to print in the board. If None, the board is
                                printed with the numbers stored in the `value_matrix` attribute, by default such matrix
                                is filled with zeros.
        """
        if colors is not None:
            assert len(colors) == 81, "Coloring must match number of vertices"
            self.value_matrix = np.array(colors).reshape(9, 9)

        double_horizontal_line = "+---------+---------+---------+"
        single_horizontal_line = "|---------+---------+---------|"

        # print the board
        print(double_horizontal_line)
        for row_index in range(9):
            # print a single horizontal line every 3 rows
            if row_index % 3 == 0 and row_index != 0:
                print(single_horizontal_line)
            # print the row (with vertical lines every 3 columns)
            row = ""
            for col_index in range(9):
                # add vertical every 3 columns
                if col_index % 3 == 0:
                    row += "|"
                # print the value of the cell, or a space if the cell is empty
                if self.value_matrix[row_index, col_index] == 0:
                    row += "   "
                else:
                    row += f" {self.value_matrix[row_index, col_index]} "
            row += "|"
            # print the row
            print(row)
        print(double_horizontal_line)

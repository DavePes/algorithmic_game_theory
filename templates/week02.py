#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """
    row_utilities = []
    row_strategies = []
    row_strategy = np.array([0.0, 1.0])
    while row_strategy[0] <= 1.0:
        best_response = np.min(row_strategy @ row_matrix) ## col_player is minimizing
        row_utilities.append(best_response)
        row_strategies.append(row_strategy[0])
        row_strategy[0] += step_size
        row_strategy[1] -= step_size

    # Plot the best response value function
    row_strategies = np.array(row_strategies)
    row_utilities = np.array(row_utilities)
    plt.plot(row_strategies, row_utilities)
    plt.xlabel('Row Strategy')
    plt.ylabel('Best Response Utility')
    plt.title('Best Response Value Function')
    plt.grid()
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support

    Returns
    -------
    np.ndarray | None
        The opponent's strategy, if it exists, otherwise `None`
    """
    raise NotImplementedError


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import combinations

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
    num_row_support = len(row_support)
    num_col_support = len(col_support)
    submatrix = matrix[np.ix_(row_support, col_support)]
    A_eq = np.zeros((num_row_support + 1, num_col_support + 1))
    b_eq = np.zeros(num_row_support + 1)
    b_eq[-1] = 1.0

    A_eq[:-1, :-1] = submatrix
    A_eq[:num_row_support,-1] = -1.0
    A_eq[-1,:num_col_support] = 1.0
    c = np.zeros(num_col_support + 1)
    epsilon = 1e-9
    bounds = [(epsilon, None)] * num_col_support + [(None, None)]
    result = linprog(c=c,A_eq = A_eq,b_eq = b_eq,bounds = bounds,method='highs')
    if result.success:
        return result.x[:-1]
    return None


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
    def is_nash_equilibrium(
    row_strategy: np.ndarray, col_strategy: np.ndarray, row_matrix: np.ndarray, col_matrix: np.ndarray
    ) -> bool:
        row_payoff = row_matrix @ col_strategy
        col_payoff = row_strategy @ col_matrix

        player_row_payoff = row_strategy @ row_payoff  
        player_col_payoff = col_payoff @ col_strategy
        if (np.max(row_payoff) > player_row_payoff + 1e-8) or (np.max(col_payoff) > player_col_payoff + 1e-8):
            return False
        return True
    equilibria = []
    num_rows = row_matrix.shape[0]
    num_cols = col_matrix.shape[1]
    for k in range(1, min(num_rows, num_cols) + 1):
        for row_supp in combinations(range(num_rows), k):
            row_supp = np.array(row_supp)
            for col_supp in combinations(range(num_cols), k):
                col_supp = np.array(col_supp)
                partial_col_strategy = verify_support(row_matrix, row_supp, col_supp)
                partial_row_strategy = verify_support(col_matrix.T, col_supp, row_supp)
                if partial_col_strategy is None or partial_row_strategy is None:
                    continue
                col_strategy = np.zeros(num_cols)
                row_strategy = np.zeros(num_rows)
                col_strategy[col_supp] = partial_col_strategy
                row_strategy[row_supp] = partial_row_strategy

                if is_nash_equilibrium(row_strategy, col_strategy, row_matrix, col_matrix):
                    equilibria.append((row_strategy, col_strategy))
    print(len(equilibria))
    return equilibria
def main() -> None:
    pass


if __name__ == '__main__':
    main()
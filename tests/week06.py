#!/usr/bin/env python3

import numpy as np

def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """

    regrets = np.maximum(regrets, 0)
    if (np.sum(regrets) == 0):
        strategy = np.ones(regrets.shape) / regrets.shape[0]
        return strategy
    else:
        return regrets / np.sum(regrets)


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """
    cum_row_regrets = np.zeros(row_matrix.shape[0])
    cum_col_regrets = np.zeros(col_matrix.shape[1])
    avg_row_strategy = np.zeros(row_matrix.shape[0])
    avg_col_strategy = np.zeros(col_matrix.shape[1])
    strategies = []
    sum_row_strategy = np.zeros(row_matrix.shape[0])
    sum_col_strategy = np.zeros(col_matrix.shape[1])
    for i in range(num_iters):
        row_strategy = regret_matching(cum_row_regrets)
        col_strategy = regret_matching(cum_col_regrets)

        row_regret = row_matrix @ col_strategy - row_strategy @ row_matrix @ col_strategy
        col_regret = col_matrix.T @ row_strategy - col_strategy @ col_matrix.T @ row_strategy


        cum_row_regrets += row_regret
        cum_col_regrets += col_regret
        sum_row_strategy += row_strategy
        sum_col_strategy += col_strategy
        avg_row_strategy = sum_row_strategy / (i + 1)
        avg_col_strategy = sum_col_strategy / (i + 1)
        strategies.append((avg_row_strategy, avg_col_strategy))
    return strategies


def main() -> None:
    pass


if __name__ == '__main__':
    main()

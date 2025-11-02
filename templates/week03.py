#!/usr/bin/env python3

import numpy as np


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """
    row_util = row_strategy @ row_matrix @ col_strategy
    best_r_util = np.max(row_matrix @ col_strategy)

    col_util = row_strategy @ col_matrix @ col_strategy
    best_c_util = np.max(row_strategy @ col_matrix)

    return np.array([best_r_util - row_util, best_c_util - col_util])


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The NashConv value of the given strategy profile
    """
    deltas = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.sum(np.maximum(deltas))


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The exploitability value of the given strategy profile
    """
    return 0.5 * compute_nash_conv(
        row_matrix, col_matrix, row_strategy, col_strategy
    )


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of iterations.

    Although any averaging method is valid, the reference solution updates the
    average strategy vectors using a moving average. Therefore, it is recommended
    to use the same averaging method to avoid numerical discrepancies during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """
    avg_row_strategy = np.ones(row_matrix.shape[0])/row_matrix.shape[0]
    avg_col_strategy = np.ones(col_matrix.shape[1])/col_matrix.shape[1]
    last_row_strategy = avg_row_strategy
    last_col_strategy = avg_col_strategy
    for i in range(num_iters):
        if (i % 2 == 0):
            if naive:
                row_strategy = np.argmax(row_matrix @ last_col_strategy)
                last_row_strategy = row_strategy
            else:
                row_strategy = np.argmax(row_matrix @ avg_col_strategy)
                avg_row_strategy = (avg_row_strategy * (i//2) + row_strategy) / (i//2 + 1)
        else:
            if naive:
                col_strategy = np.argmax(last_row_strategy @ col_matrix)
                last_col_strategy = col_strategy
            else:
                col_strategy = np.argmax(avg_row_strategy @ col_matrix)
                avg_col_strategy = (avg_col_strategy * (i//2) + col_strategy) / (i//2 + 1)
    return [(avg_row_strategy, avg_col_strategy)]


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[np.float64]
        A sequence of exploitability values, one for each strategy profile
    """
    exploitabilities = []
    for strategy in strategies:
        exploitability = compute_exploitability(row_matrix, col_matrix, strategy[0], strategy[1])
        exploitabilities.append(exploitability)
    return exploitabilities


def main() -> None:
    pass


if __name__ == '__main__':
    main()

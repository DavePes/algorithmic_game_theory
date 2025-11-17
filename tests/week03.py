#!/usr/bin/env python3

import numpy as np
import week01

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
) -> float:
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
    float
        The NashConv value of the given strategy profile
    """
    deltas = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.sum(deltas)


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> float:
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
    float
        The exploitability value of the given strategy profile
    """
    return 0.5 * compute_nash_conv(
        row_matrix, col_matrix, row_strategy, col_strategy
    )


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of epochs.

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
    m, n = row_matrix.shape
    assert col_matrix.shape == (m, n)
    history = []

    avg_row = np.ones(m,dtype=np.float64) / m
    avg_col = np.ones(n, dtype=np.float64) / n
    row_action = week01.calculate_best_response_against_col(row_matrix, avg_col)
    col_action = week01.calculate_best_response_against_row(col_matrix, avg_row)
    avg_row = row_action.copy()
    avg_col = col_action.copy()
    last_col = col_action
    last_row = row_action
    history.append((avg_row.copy(), avg_col.copy()))
    for i in range(2, num_iters + 1):
        # row moves
        target_col = last_col if naive else avg_col
        row_action = week01.calculate_best_response_against_col(row_matrix, target_col)
        target_row = last_row if naive else avg_row
        col_action = week01.calculate_best_response_against_row(col_matrix, target_row)

        avg_row += (row_action - avg_row) / i
        avg_col += (col_action - avg_col) / i

        last_row = row_action
        last_col = col_action
        history.append((avg_row.copy(), avg_col.copy()))
    return history

    

def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[float]:
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
    list[float]
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

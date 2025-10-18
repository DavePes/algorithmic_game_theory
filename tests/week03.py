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

    avg_row = np.ones(m) / m
    avg_col = np.ones(n) / n
    last_row = avg_row.copy()
    last_col = avg_col.copy()

    row_count = 0
    col_count = 0

    for i in range(num_iters):
        if i % 2 == 0:
            # row moves
            target_col = last_col if naive else avg_col
            br_idx = np.argmax(row_matrix @ target_col)
            row_pure = np.zeros(m); row_pure[br_idx] = 1
            last_row = row_pure
            row_count += 1
            avg_row = (avg_row * (row_count - 1) + row_pure) / row_count
        else:
            # column moves
            target_row = last_row if naive else avg_row
            br_idx = np.argmax(target_row @ col_matrix)
            col_pure = np.zeros(n); col_pure[br_idx] = 1
            last_col = col_pure
            col_count += 1
            avg_col = (avg_col * (col_count - 1) + col_pure) / col_count

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

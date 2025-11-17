#!/usr/bin/env python3

import numpy as np
import week01 
import week04

def double_oracle(
    row_matrix: np.ndarray, eps: float, rng: np.random.Generator
) -> tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
    """Run Double Oracle until a termination condition is met.

    The reference implementation generates the initial restricted game by
    randomly sampling one pure action for each player using `rng.integers`.

    The algorithm terminates when either:
        1. the difference between the upper and the lower bound on the game value drops below `eps`
        2. both players' best responses are already contained in the current restricted game

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    eps : float
        The required accuracy for the approximate Nash equilibrium
    rng : np.random.Generator
        A random number generator

    Returns
    -------
    tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]
        A tuple containing a sequence of strategy profiles and a sequence of corresponding supports
    """
    strategies = []
    supports = []
    col_matrix = -row_matrix.T
    m,n = row_matrix.shape
    X = rng.integers(m, size=1).tolist()
    Y = rng.integers(n, size=1).tolist()
    while True:
        restricted = row_matrix[np.ix_(X, Y)]
        row_strategy,col_strategy = week04.find_nash_equilibrium(restricted)

        row_full = np.zeros(m)
        row_full[X] = row_strategy
        row_strategy = row_full

        col_full = np.zeros(n)
        col_full[Y] = col_strategy
        col_strategy = col_full

        x = week01.calculate_best_response_against_col(row_matrix,col_strategy)
        x_pure = np.argmax(x)
        y = week01.calculate_best_response_against_row(col_matrix,row_strategy)
        y_pure = np.argmax(y)
        strategies.append((row_strategy, col_strategy))
        supports.append((np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)))
        if (x_pure in X) and (y_pure in Y):
            return (strategies, supports)
        if (x_pure not in X):
            X.append(x_pure)
        if (y_pure not in Y):
            Y.append(y_pure)
        u_row = week01.evaluate_zero_sum(row_matrix,row_strategy,y)
        u_col = week01.evaluate_zero_sum(row_matrix,x,col_strategy)
        if abs(u_col[0]-u_row[0]) <= eps:
            return (strategies, supports)
    

def main() -> None:
    pass


if __name__ == '__main__':
    main()

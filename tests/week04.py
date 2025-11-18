#!/usr/bin/env python3

import numpy as np
import week02 
from scipy.optimize import linprog
def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """
    def find_strategy(matrix: np.ndarray) -> tuple[np.ndarray, float]:
        # Objective: minimize -u
        m, n = matrix.shape
        c = np.zeros(m + 1)
        c[-1] = -1.0
        A_ub = np.ones((n, m + 1))
        A_ub[:,:-1] = -matrix.T
        b_ub = np.zeros(n)

        A_eq = np.ones((1, m + 1))
        A_eq[0,-1] = 0.0

        b_eq = np.array([1.0])
        bounds = [(0, 1)] * m + [(None, None)]
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        p = result.x[:-1]
        u = result.x[-1]
        return result.x[:-1], result.x[-1]

    row_strategy, game_value_row = find_strategy(row_matrix)
    col_strategy, game_value_col = find_strategy(-row_matrix.T)
    assert abs(game_value_row + game_value_col) < 1e-8, "Game values do not match!"
    return row_strategy, col_strategy

def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players’ utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """
    num_row_actions, num_col_actions = row_matrix.shape
    ## Row inequalities first row
    A_row = np.zeros(((num_row_actions-1)*num_row_actions,num_col_actions*num_row_actions),dtype=np.float64)
    for ai in range(num_row_actions):
        for prime_a in range(num_row_actions):
            if ai == prime_a:
                continue
            row_index = ai*(num_row_actions-1)+prime_a-(1 if prime_a>ai else 0)
            col_index = ai*num_col_actions
            A_row[row_index, col_index:(col_index+num_col_actions)] = row_matrix[ai,:] - row_matrix[prime_a,:]
    ## Column inequalities second row
    A_col = np.zeros(((num_col_actions-1)*num_col_actions,num_col_actions*num_row_actions),dtype=np.float64)
    for aj in range(num_col_actions):
        for prime_a in range(num_col_actions):
            if aj == prime_a:
                continue
            row_index = aj*(num_col_actions-1)+prime_a-(1 if prime_a>aj else 0)
            col_index = aj
            A_col[row_index, col_index:(col_index+num_col_actions*num_row_actions):num_col_actions] = col_matrix[:,aj] - col_matrix[:,prime_a]
    
    # Inequalities in <= form
    A_ub = -np.vstack([A_row, A_col])   # flip sign to get <= 0
    b_ub = np.zeros(A_ub.shape[0], dtype=np.float64)

    # Probability simplex
    A_eq = np.ones((1, num_row_actions*num_col_actions), dtype=np.float64)
    b_eq = np.array([1.0], dtype=np.float64)

    # Variables: p_{ij} flattened row-major
    c = np.zeros(num_row_actions*num_col_actions, dtype=np.float64)
    bounds = [(0.0, 1)] * (num_row_actions*num_col_actions)

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return res.x.reshape((num_row_actions, num_col_actions))
    else:
        raise ValueError("Linear program to find correlated equilibrium failed.")


def main() -> None:
    pass


if __name__ == '__main__':
    main()

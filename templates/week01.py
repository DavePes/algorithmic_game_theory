#!/usr/bin/env python3

import numpy as np

def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """
    # first player is row, second is col
    first_player = row_strategy @ row_matrix @ col_strategy
    second_player = row_strategy @ col_matrix @ col_strategy
    exp_utils = np.array([first_player, second_player])
    
    return exp_utils


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    first_player = row_strategy @ row_matrix @ col_strategy
    second_player = -first_player
    exp_utils = np.array([first_player, second_player])
    return exp_utils


def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """
    best_response = row_strategy @ col_matrix
    col_strategy = np.zeros(col_matrix.shape[1])
    col_strategy[np.argmax(best_response)] = 1  
    return col_strategy


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """
    best_response = row_matrix @ col_strategy
    row_strategy = np.zeros(row_matrix.shape[0])
    row_strategy[np.argmax(best_response)] = 1
    return row_strategy


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float64
        The expected utility of the row player
    """
    col_strategy = np.zeros(col_matrix.shape[1])
    best_response_index = np.argmax(calculate_best_response_against_row(col_matrix, row_strategy))
    col_strategy[best_response_index] = 1
    return row_strategy @ row_matrix @ col_strategy


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The expected utility of the column player
    """
    row_strategy = np.zeros(row_matrix.shape[0])
    best_response_index = np.argmax(calculate_best_response_against_col(row_matrix, col_strategy))
    row_strategy[best_response_index] = 1
    return row_strategy @ col_matrix @ col_strategy


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """
    dominated_actions = []
    for a in range(matrix.shape[0]):
        for b in range(matrix.shape[0]):
            if a != b and all(matrix[a, :] < matrix[b, :]):
                dominated_actions.append(a)
                break
    return np.array(dominated_actions)



def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """
    row_actions = np.arange(row_matrix.shape[0])
    col_actions = np.arange(col_matrix.shape[1])
    while True:
        changed = False
        row_dominated_actions = find_strictly_dominated_actions(row_matrix)
        col_dominated_actions = find_strictly_dominated_actions(col_matrix.T)
        if (len(row_dominated_actions) != 0):
            num_rows = row_matrix.shape[0]
            mask = np.full(num_rows,True)
            mask[row_dominated_actions] = False
            row_matrix = row_matrix[mask]
            col_matrix = col_matrix[mask]
            row_actions = row_actions[mask]
            changed = True
        if (len(col_dominated_actions) != 0):
            num_cols = col_matrix.shape[1]
            mask = np.full(num_cols,True)
            mask[col_dominated_actions] = False
            row_matrix = row_matrix[:,mask]
            col_matrix = col_matrix[:,mask]
            col_actions = col_actions[mask]
            changed = True
        if not changed:
            break
    return row_matrix, col_matrix, row_actions, col_actions


def main() -> None:
    pass


if __name__ == '__main__':
    main()

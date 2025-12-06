#!/usr/bin/env python3
import numpy as np
import copy

"""
Extensive-form games assignments.

Starting this week, the templates will no longer contain exact function
signatures and there will not be any automated tests like we had for the
normal-form games assignments. Instead, we will provide sample outputs
produced by the reference implementations which you can use to verify
your solutions. The reason for this change is that there are many valid
ways to represent game trees (e.g. flat array-based vs. pointer-based),
information sets and strategies in extensive-form games. Figuring out
the most suitable representations is an important part of assignments
in this block. Unfortunately, this freedom makes automated testing
pretty much impossible.
"""

class nodes():
    def __init__(self, state, history):
        self.state = state
        self.history = history
        self.children = {}

class groupped_nodes():
    def __init__(self, state,history):
        self.states = [state]
        self.history = history
        self.state = state
        self.total_value = {}
        

def get_history_of_last_player(observed_moves,num_players,observing_player):
    observed_moves_of_last_player = []
    for i, h in enumerate(observed_moves):
        if h[0] == -1 and i % num_players == observing_player or h[0] != -1:
            observed_moves_of_last_player.append(h)
        else:
            observed_moves_of_last_player.append((-1, -1))
    return observed_moves_of_last_player


def traverse_tree(env, state,num_players,groupped_nodes_dict,observed_moves=[]):
    """Build a full extensive-form game tree for a given game."""
    if state.terminated or state.truncated:
        return nodes(state, observed_moves)
    
    if (len(observed_moves) > 0 and not state.is_chance_node):
        
        observed_moves_of_last_player = get_history_of_last_player(observed_moves,num_players,state.current_player.item())
        
        if tuple(observed_moves_of_last_player) not in groupped_nodes_dict:
            groupped_nodes_dict[tuple(observed_moves_of_last_player)] = groupped_nodes(state, observed_moves_of_last_player)
        else:
            groupped_nodes_dict[tuple(observed_moves_of_last_player)].states.append(state)
    node = nodes(state, observed_moves)
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:    
            env_copy = copy.deepcopy(env)
            actor = -1 if state.is_chance_node else state.current_player.item()

            node.children[action] = traverse_tree(env_copy, env_copy.step(state, action), num_players, groupped_nodes_dict, observed_moves + [(actor, action)])        
    return node

def evaluate(*args, **kwargs):
    """Compute the expected utility of each player in an extensive-form game."""

    raise NotImplementedError


def compute_best_response(node,opponent_strategy,num_players,player_id,groupped_nodes,probability,observed_moves = []):
    """Compute a best response strategy for a given player against a fixed opponent's strategy."""
    state = node.state

    if state.terminated or state.truncated:
        return state.rewards[player_id]
    
    expected_utility = {}
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:
            actor = -1 if state.is_chance_node else state.current_player.item()
            observed_moves_of_last_player = get_history_of_last_player(observed_moves,num_players,state.current_player.item())

            if state.is_chance_node:
                expected_utility[action] = state.chance_strategy[action] * compute_best_response(node.children[action],opponent_strategy,num_players,player_id,groupped_nodes,probability * state.chance_strategy[action],observed_moves + [(actor, action)])
            elif state.current_player.item() == player_id:
                expected_utility[action] = compute_best_response(node.children[action],opponent_strategy,num_players,player_id,groupped_nodes,probability,observed_moves + [(actor, action)])
                groupped_nodes[tuple(observed_moves_of_last_player)].total_value[action] = groupped_nodes[tuple(observed_moves_of_last_player)].total_value.get(action, 0) + expected_utility[action] * probability
            elif state.current_player.item() != player_id:
                opponent_prob = opponent_strategy[observed_moves_of_last_player][action]
                expected_utility[action] = opponent_prob * compute_best_response(node.children[action],opponent_strategy,num_players,player_id,groupped_nodes,probability*opponent_prob,observed_moves + [(actor, action)])
    if state.current_player.item() == player_id:
        return max(expected_utility.values())
    return sum(expected_utility.values())

def compute_average_strategy(*args, **kwargs):
    """Compute a weighted average of a pair of behavioural strategies for a given player."""

    raise NotImplementedError


def compute_exploitability(*args, **kwargs):
    """Compute and plot the exploitability of a sequence of strategy profiles."""

    raise NotImplementedError


def main() -> None:
    from kuhn_poker import KuhnPokerNumpy as KuhnPoker

    # The implementation of the game is a part of a JAX library called `pgx`.
    # You can find more information about it here: https://www.sotets.uk/pgx/kuhn_poker/
    # We wrap the original implementation to add an explicit chance player and convert
    # everything from JAX arrays to Numpy arrays. There's also a JAX version which you
    # can import using `from kuhn_poker import KuhnPoker` if interested ;)
    env = KuhnPoker()

    # Initialize the environment with a random seed
    state = env.init(0)
    groupped_nodes_dict = {}
    #game_tree = traverse_tree(env, state, 2, groupped_nodes_dict)

    while not (state.terminated or state.truncated):
        if state.is_chance_node:
            uniform_strategy = state.legal_action_mask / np.sum(state.legal_action_mask)
            assert np.allclose(state.chance_strategy, uniform_strategy), (
                'The chance strategy is not uniform!'
            )

        # Pick the first legal action
        action = np.argmax(state.legal_action_mask)

        # Take a step in the environment
        state = env.step(state, action)

    assert np.sum(state.rewards) == 0, 'The game is not zero-sum!'
    assert state.terminated or state.truncated, 'The game is not over!'


if __name__ == '__main__':
    main()

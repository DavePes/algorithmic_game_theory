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
    def __init__(self, state, history,information_set=None):
        self.state = state
        self.history = history
        self.children = {}
        self.information_set = information_set
class information_set():
    def __init__(self, state,history):
        self.states = [state]
        self.history = history
        self.state = state
        self.total_value = {}
        self.player_id = state.current_player.item() if not state.is_chance_node else -1

    
def get_history_of_last_player(observed_moves,num_players,observing_player):
    observed_moves_of_last_player = []
    for i, h in enumerate(observed_moves):
        if h[0] == -1 and i % num_players == observing_player or h[0] != -1:
            observed_moves_of_last_player.append(h)
        else:
            observed_moves_of_last_player.append((-1, -1))
    return observed_moves_of_last_player


def traverse_tree(env, state,num_players,information_set_dict,observed_moves=[]):
    """Build a full extensive-form game tree for a given game."""
    if state.terminated or state.truncated:
        return nodes(state, observed_moves)
    
    if (len(observed_moves) > 0 and not state.is_chance_node):
        
        observed_moves_of_last_player = get_history_of_last_player(observed_moves,num_players,state.current_player.item())
        
        if tuple(observed_moves_of_last_player) not in information_set_dict:
            information_set_dict[tuple(observed_moves_of_last_player)] = information_set(state, observed_moves_of_last_player)
        else:
            information_set_dict[tuple(observed_moves_of_last_player)].states.append(state)
    information_set_of_last_player = None if state.is_chance_node and len(observed_moves) == 0 else information_set_dict[tuple(get_history_of_last_player(observed_moves,num_players,state.current_player.item()))]
    node = nodes(state, observed_moves,information_set_of_last_player)
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:    
            env_copy = copy.deepcopy(env)
            actor = -1 if state.is_chance_node else state.current_player.item()

            node.children[action] = traverse_tree(env_copy, env_copy.step(state, action), num_players, information_set_dict, observed_moves + [(actor, action)])        
    return node

def evaluate(node, strategy_profile, num_players):
    """Compute the expected utility of each player in an extensive-form game."""
    state = node.state
    utilities = {player_id: 0 for player_id in range(num_players)}
    if state.terminated or state.truncated:
        return state.rewards
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:
            prob = 0
            if state.is_chance_node:
                prob = state.chance_strategy[action]
            else:
                prob = strategy_profile[node.information_set.history][action] 

            if prob >= 0:
                child_utilities = evaluate(node.children[action], strategy_profile, num_players)
                for player_id in range(num_players):
                    utilities[player_id] += prob * child_utilities[player_id]
    return utilities

def reset_information_set_values(information_set_dict):
    for key in information_set_dict:
        information_set_dict[key].total_value = {}

def compute_best_response(node,strategy_profile,num_players,player_id,information_set,probability):
    """Compute a best response strategy for a given player against a fixed opponent's strategy."""
    def best_response_for_each_set(node,strategy_profile,num_players,player_id,probability):
        state = node.state

        if state.terminated or state.truncated:
            return state.rewards[player_id]
        expected_utility = {}
        for action,legal_action in enumerate(state.legal_action_mask):
            if legal_action == True:
                observed_moves_of_last_player = node.information_set.history
                if state.is_chance_node:
                    expected_utility[action] = state.chance_strategy[action] * best_response_for_each_set(node.children[action],strategy_profile,num_players,player_id,information_set,probability * state.chance_strategy[action])
                elif state.current_player.item() == player_id:
                    expected_utility[action] = best_response_for_each_set(node.children[action],strategy_profile,num_players,player_id,information_set,probability)
                    information_set[observed_moves_of_last_player].total_value[action] = information_set[observed_moves_of_last_player].total_value.get(action, 0) + expected_utility[action] * probability
                elif state.current_player.item() != player_id:
                    opponent_prob = strategy_profile[observed_moves_of_last_player][action]
                    expected_utility[action] = opponent_prob * best_response_for_each_set(node.children[action],strategy_profile,num_players,player_id,information_set,probability*opponent_prob)
        if state.current_player.item() == player_id:
            return max(expected_utility.values())
        return sum(expected_utility.values())
        
    reset_information_set_values(information_set)
    best_response_for_each_set(node,strategy_profile,num_players,player_id,probability)

    best_response = {}
    for groupped_node in information_set.keys():
        if information_set[groupped_node].player_id != player_id:
            best_response[groupped_node] = strategy_profile[groupped_node]
            continue
        best_action = -1
        best_value = -float('inf')
        best_response[groupped_node] = {}
        for action, value in information_set[groupped_node].total_value.items():
            best_response[groupped_node][action] = 0.0
            if value > best_value:
                best_value = value
                best_action = action
        best_response[groupped_node][best_action] = 1.0
    return best_response

        


def compute_average_strategy(node,strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1,prob_strategy_2,new_strategy):
    """Compute a weighted average of a pair of behavioural strategies for a given player."""
    def compute_strategy(node,strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1,prob_strategy_2,new_strategy):
    
        state = node.state
        if state.terminated or state.truncated:
            return
        for action,legal_action in enumerate(state.legal_action_mask):
            if legal_action == True:
                observed_moves_of_last_player = node.information_set.history
                if state.is_chance_node:
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1 * state.chance_strategy[action],prob_strategy_2 * state.chance_strategy[action],new_strategy)
                elif state.current_player.item() == player_id:
                    if new_strategy.get(observed_moves_of_last_player) is None:
                        new_strategy[observed_moves_of_last_player] = {}
                    prob_1 = strategy_1[observed_moves_of_last_player][action]
                    prob_2 = strategy_2[observed_moves_of_last_player][action]
                    new_prob = weight_1 * prob_strategy_1 * prob_1 + weight_2 * prob_strategy_2 * prob_2
                    new_strategy[observed_moves_of_last_player][action] = new_strategy[observed_moves_of_last_player].get(action, 0) + new_prob
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1*prob_1,prob_strategy_2*prob_2,new_strategy)
                elif state.current_player.item() != player_id:
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1,prob_strategy_2,new_strategy)
    compute_strategy(node,strategy_1,strategy_2,weight_1,weight_2,num_players,player_id,prob_strategy_1,prob_strategy_2,new_strategy)
    ## Normalize the new strategy
    for key in new_strategy:
        total = sum(new_strategy[key].values())
        if total == 0:
            continue
        for action in new_strategy[key]:
            new_strategy[key][action] /= total
    


def compute_exploitability(node,information_set,strategy_profile, num_players):
    """Compute and plot the exploitability of a sequence of strategy profiles."""
    ## compute deltas
    utilities = evaluate(node, strategy_profile, num_players)
    best_utilities = {}
    for player_id in range(num_players):
        best_response = compute_best_response(node, strategy_profile, num_players, player_id, information_set, 1.0)
        best_utilities[player_id] = evaluate(node,best_response, num_players)[player_id]
    deltas = [best_utilities[player_id] - utilities[player_id] for player_id in range(num_players)]
    return sum(deltas)*0.5

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
    information_set_dict = {}
    game_tree = traverse_tree(env, state, 2, information_set_dict)


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

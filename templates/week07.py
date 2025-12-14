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
global_history = []

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
        self.mapped_history = None

    
def get_history_of_last_player(observed_moves,num_players,observing_player):
    observed_moves_of_last_player = []
    for i, h in enumerate(observed_moves):
        if i % num_players == observing_player or h[0] != -1:
            observed_moves_of_last_player.append(h)
        else:
            observed_moves_of_last_player.append((-1, -1))
    return observed_moves_of_last_player

def map_history(observed_moves,player_id):
    ## only works for kuhn_poker
    ## it is like this
    # cards 0,1,2
    # J,Q,K
    # or
    # check bet
    # fold call
    chance_nodes = ["J","Q","K"]
    has_bet = False
    check_bet = ["Bet","Check"]
    fold_call = ["Call","Fold"]
    mapped_history = []
    mapped_action = ""
    for i, h in enumerate(observed_moves):
        # h[0] = -1 is chance node
        if h[0] == -1:
            mapped_action = chance_nodes[h[1]]
        elif h[1] != -1:
            if has_bet == False:
                mapped_action = check_bet[h[1]]
                if check_bet[h[1]] == "Bet":
                    has_bet = True
            else:
                mapped_action = fold_call[h[1]]
        if i % 2 == player_id or h[0] != -1:
            mapped_history.append(mapped_action)
        else:
            mapped_history.append("")
    global_history.append(tuple(mapped_history))
    return tuple(mapped_history)
            
        
def traverse_tree(env, state,num_players,information_set_dict,observed_moves=[],map_kuhn_poker=False):
    """Build a full extensive-form game tree for a given game."""
    if state.terminated or state.truncated:
        return nodes(state, observed_moves)
    
    if not state.is_chance_node:
        observed_moves_of_last_player = get_history_of_last_player(observed_moves,num_players,state.current_player.item())
        if tuple(observed_moves_of_last_player) not in information_set_dict:
            information_set_dict[tuple(observed_moves_of_last_player)] = information_set(state, tuple(observed_moves_of_last_player))
            information_set_dict[tuple(observed_moves_of_last_player)].mapped_history = map_history(observed_moves, state.current_player.item()) if map_kuhn_poker else None
        else:
            information_set_dict[tuple(observed_moves_of_last_player)].states.append(state)
    information_set_of_last_player = None if state.is_chance_node else information_set_dict[tuple(get_history_of_last_player(observed_moves,num_players,state.current_player.item()))]
    node = nodes(state, observed_moves,information_set_of_last_player)
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:    
            env_copy = copy.deepcopy(env)
            actor = -1 if state.is_chance_node else state.current_player.item()

            node.children[action] = traverse_tree(env_copy, env_copy.step(state, action), num_players, information_set_dict, observed_moves + [(actor, action)],map_kuhn_poker)        
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
                prob = strategy_profile[state.current_player.item()][node.information_set.history][action] 

            child_utilities = evaluate(node.children[action], strategy_profile, num_players)
            for player_id in range(num_players):
                utilities[player_id] += prob * child_utilities[player_id]
    return utilities

def reset_information_set_values(information_set_dict):
    for key in information_set_dict:
        information_set_dict[key].total_value = {}

def compute_best_response(node,opponents_strategy_profile,player_id,information_set,probability):
    """Compute a best response strategy for a given player against a fixed opponent's strategy."""
    def best_response_for_each_set(node,opponents_strategy_profile,player_id,information_set,probability):
        state = node.state
        current_player = state.current_player.item()
        if state.terminated or state.truncated:
            return state.rewards[player_id]
        expected_utility = {}
        for action,legal_action in enumerate(state.legal_action_mask):
            if legal_action == True:
                observed_moves_of_last_player = None if state.is_chance_node or len(node.information_set.history) == 0 else node.information_set.history
                if state.is_chance_node:
                    expected_utility[action] = state.chance_strategy[action] * best_response_for_each_set(node.children[action],opponents_strategy_profile,player_id,information_set,probability * state.chance_strategy[action])
                elif current_player == player_id:
                    expected_utility[action] = best_response_for_each_set(node.children[action],opponents_strategy_profile,player_id,information_set,probability)
                    information_set[observed_moves_of_last_player].total_value[action] = information_set[observed_moves_of_last_player].total_value.get(action, 0) + expected_utility[action] * probability
                elif current_player != player_id:
                    opponent_prob = opponents_strategy_profile[current_player][observed_moves_of_last_player][action]
                    expected_utility[action] = opponent_prob * best_response_for_each_set(node.children[action],opponents_strategy_profile,player_id,information_set,probability*opponent_prob)
        if current_player == player_id:
            return max(expected_utility.values())
        return sum(expected_utility.values())
        
    reset_information_set_values(information_set)
    best_response_for_each_set(node,opponents_strategy_profile,player_id,information_set,probability)

    best_response = {player_id: {}}
    for groupped_node in information_set.keys():
        if information_set[groupped_node].player_id != player_id:
            continue
        best_action = -1
        best_value = -float('inf')
        best_response[player_id][groupped_node] = {}
        for action, value in information_set[groupped_node].total_value.items():
            best_response[player_id][groupped_node][action] = 0.0
            if value > best_value:
                best_value = value
                best_action = action
        best_response[player_id][groupped_node][best_action] = 1.0
    return best_response

        


def compute_average_strategy(node,strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1,prob_2):
    averaged_strategy = {}
    """Compute a weighted average of a pair of behavioural strategies for a given player."""
    def compute_strategy(node,strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1,prob_2,averaged_strategy):
    
        state = node.state
        if state.terminated or state.truncated:
            return
        for action,legal_action in enumerate(state.legal_action_mask):
            if legal_action == True:
                observed_moves_of_last_player = None if state.is_chance_node or len(node.information_set.history) == 0 else node.information_set.history
                if state.is_chance_node:
                    # Propagate the chance probability equally to both strategy paths
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1 * state.chance_strategy[action],prob_2 * state.chance_strategy[action],averaged_strategy)
                elif state.current_player.item() == player_id:
                    if averaged_strategy.get(observed_moves_of_last_player) is None:
                        averaged_strategy[observed_moves_of_last_player] = {}
                    # Retrieve local probabilities from both strategies
                    prob_1_action = strategy_1[player_id][observed_moves_of_last_player][action]
                    prob_2_action = strategy_2[player_id][observed_moves_of_last_player][action]
                    #Weight the local probability by the reach probability
                    # This accounts for the fact that one strategy might visit this node much more often than the other.
                    new_prob = weight_1 * prob_1 * prob_1_action + weight_2 * prob_2 * prob_2_action
                    averaged_strategy[observed_moves_of_last_player][action] = averaged_strategy[observed_moves_of_last_player].get(action, 0) + new_prob
                    ## DEBUG PRINT
                    # if observed_moves_of_last_player == ((-1, 1), (-1, -1)):  #[('Q', ''), ('Bet', 'Check')]
                    #     print(f"Action: {action}, prob_1_action = {prob_1_action}, prob_2_action = {prob_2_action}, weight_1 = {weight_1}, weight_2 = {weight_2}, prob_1 = {prob_1}, prob_2 = {prob_2}")
                    #     print(f"New prob for action {action} at info set {new_prob}")
                    #     x =  5 
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1*prob_1_action,prob_2*prob_2_action,averaged_strategy)
                elif state.current_player.item() != player_id:
                    compute_strategy(node.children[action],strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1,prob_2,averaged_strategy)
    compute_strategy(node,strategy_1,strategy_2,weight_1,weight_2,player_id,prob_1,prob_2,averaged_strategy)
    ## Normalize the new strategy
    for key in averaged_strategy:
        total = sum(averaged_strategy[key].values())
        if total == 0:
            continue
        for action in averaged_strategy[key]:
            averaged_strategy[key][action] /= total
    return {player_id:averaged_strategy}
    


def compute_exploitability(node,information_set,strategy_profile, num_players):
    """Compute and plot the exploitability of a sequence of strategy profiles."""
    ## compute deltas
    utilities = evaluate(node, strategy_profile, num_players)
    deltas = []
    for player_id in range(num_players):
        # 1. Get the Best Response strategy dictionary for the current player
        best_response_strat = compute_best_response(node, strategy_profile, player_id, information_set, 1.0)
        
        # 2. Create a full profile: Current player uses BR, opponents use the original strategy_profile
        # We merge them (best_response_strat overwrites the entry for player_id)
        br_profile = {**strategy_profile, **best_response_strat}
        
        # 3. Evaluate the new profile to get the utility of the Best Response
        br_utility = evaluate(node, br_profile, num_players)[player_id]
        
        # 4. Calculate the gain (delta)
        deltas.append(br_utility - utilities[player_id])
        
    return sum(deltas) * 0.5

def uniform_strategy(player_id,information_set_dict):
    """Compute the uniform random strategy for a given player."""
    uniform_strat = {}
    for key in information_set_dict:
        if information_set_dict[key].player_id != player_id:
            continue
        uniform_strat[key] = {}
        legal_actions = [action for action, legal in enumerate(information_set_dict[key].states[0].legal_action_mask) if legal]
        prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            uniform_strat[key][action] = prob
    return uniform_strat


def print_stats(iteration, avg_strategy_1, avg_strategy_2, best_response_1, best_response_2, game_tree, information_set_dict):
    mapped_histories= [
    [('J', ''), ("Bet", "Check")],
    [('Q', ''), ("Bet", "Check")],
    [('K', ''), ("Bet", "Check")],
    [('J', '', 'Check', 'Bet'), ("Call", "Fold")],
    [('Q', '', 'Check', 'Bet'), ("Call", "Fold")],
    [('K', '', 'Check', 'Bet'), ("Call", "Fold")],
    [('', 'J', 'Check'), ("Bet", "Check")],
    [('', 'J', 'Bet'), ("Call", "Fold")],
    [('', 'Q', 'Check'), ("Bet", "Check")],
    [('', 'Q', 'Bet'), ("Call", "Fold")],
    [('', 'K', 'Check'), ("Bet", "Check")],
    [('', 'K', 'Bet'), ("Call", "Fold")],
]
    curr_profile = {**avg_strategy_1, **avg_strategy_2}
    utils = evaluate(game_tree, curr_profile, 2)
    print(f"Iter {iteration}: Utility of avg. strategies: {utils[0]:.5f}, {utils[1]:.5f}")
    for i,strat in enumerate([avg_strategy_1, avg_strategy_2, best_response_1, best_response_2]):
        for mapped_history in mapped_histories:
            for key in strat[i % 2]:
                if information_set_dict[key].mapped_history == mapped_history[0]:
                    prob_action_1 = strat[i % 2][key][0]
                    prob_action_2 = strat[i % 2][key][1]
                    if i < 2:
                        print(f"Iter {iteration}: Avg. strategy of P{(i % 2) + 1} at {mapped_history[0]}: {mapped_history[1][0]}: {prob_action_1:.5f}, {mapped_history[1][1]}: {prob_action_2:.5f}")
                    if i >=2:
                        print(f"Iter {iteration}: BR of P{(i % 2) + 1} against P{((i + 1) % 2) + 1}'s avg. strategy at {mapped_history[0]}: {mapped_history[1][0]}: {prob_action_1:.5f}, {mapped_history[1][1]}: {prob_action_2:.5f}")
    print("")
def fictious_play(start_node, num_iters, information_set_dict):
    history = []
    
    # Initialization
    avg_strategy_1 = {0: uniform_strategy(0, information_set_dict)}
    avg_strategy_2 = {1: uniform_strategy(1, information_set_dict)}
    
    # Initial Best Responses
    best_response_1 = compute_best_response(start_node, avg_strategy_2, 0, information_set_dict, 1.0)
    best_response_2 = compute_best_response(start_node, avg_strategy_1, 1, information_set_dict, 1.0)
    
    print_stats(1, avg_strategy_1, avg_strategy_2, best_response_1, best_response_2, start_node,information_set_dict)
    avg_strategy_1 = compute_average_strategy(start_node, avg_strategy_1, best_response_1, 1, 1, 0, 1.0, 1.0)
    avg_strategy_2 = compute_average_strategy(start_node, avg_strategy_2, best_response_2, 1, 1, 1, 1.0, 1.0)
    history.append((copy.deepcopy(avg_strategy_1), copy.deepcopy(avg_strategy_2)))
    
    # Loop for remaining iterations
    for i in range(2, num_iters + 1):


        best_response_1 = compute_best_response(start_node, avg_strategy_2, 0, information_set_dict, 1.0)
        best_response_2 = compute_best_response(start_node, avg_strategy_1, 1, information_set_dict, 1.0)
        

        # Print Stats
        print_stats(i, avg_strategy_1, avg_strategy_2, best_response_1, best_response_2, start_node,information_set_dict)

        avg_strategy_1 = compute_average_strategy(start_node, avg_strategy_1, best_response_1, i, 1, 0, 1.0, 1.0)
        avg_strategy_2 = compute_average_strategy(start_node, avg_strategy_2, best_response_2, i, 1, 1, 1.0, 1.0)
        history.append((copy.deepcopy(avg_strategy_1), copy.deepcopy(avg_strategy_2)))

    return history

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
    game_tree = traverse_tree(env, state, 2, information_set_dict,[],True)
    # global_history.sort(reverse=True)
    # for gh in global_history:
    #     print(gh)
    fictious_play_history = fictious_play(game_tree, 200,information_set_dict)

    print("-" * 60)
    print(f"{'Iteration':<15} | {'Exploitability':<20}")
    print("-" * 60)

    # Iterate through history to calculate and print exploitability
    for i, (strat_1, strat_2) in enumerate(fictious_play_history):
        # Merge the two players' strategies into a single profile
        full_strategy_profile = {**strat_1, **strat_2}
        
        expl = compute_exploitability(game_tree, information_set_dict, full_strategy_profile, 2)
        print(f"{i + 1:<15} | {expl:.6f}")

    print("-" * 60)
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

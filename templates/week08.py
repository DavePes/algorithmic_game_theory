#!/usr/bin/env python3

import numpy as np
import itertools
import week07

def convert_to_normal_form(game_tree, information_set_dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert an extensive-form game into an equivalent normal-form representation.

    Feel free to conceptually split this function into smaller functions that compute
        - the set of pure strategies for a player, e.g. `_collect_pure_strategies`
        - the expected utility of a pure strategy profile, e.g. `_compute_expected_utility`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of payoff matrices for the two players in the resulting normal-form game.
    """
    p1_info_sets = []
    p2_info_sets = []
    

    for info_set in information_set_dict.values():
        if info_set.player_id == 0:
            p1_info_sets.append(info_set)
        elif info_set.player_id == 1:
            p2_info_sets.append(info_set)
    pure_strategies_1, info_sets_index_1 = _collect_pure_strategies(p1_info_sets)
    pure_strategies_2, info_sets_index_2 = _collect_pure_strategies(p2_info_sets)  
    populate_matrixes = compute_exp_util(game_tree, pure_strategies_1, info_sets_index_1, pure_strategies_2, info_sets_index_2)
    return (populate_matrixes[:,:,0], populate_matrixes[:,:,1])



def compute_exp_util(node, pure_strategies_1,info_sets_index_1, pure_strategies_2,info_sets_index_2):
    state = node.state
    if state.terminated or state.truncated:
        reward_matrix = np.zeros((len(pure_strategies_1), len(pure_strategies_2), 2))
        reward_matrix[:,:,0] = state.rewards[0]
        reward_matrix[:,:,1] = state.rewards[1]
        return reward_matrix
    # we need to accumulate utilities from all children
    current_node_utilities = 0
    for action,legal_action in enumerate(state.legal_action_mask):
        if legal_action == True:
            child_utilities = compute_exp_util(node.children[action], pure_strategies_1,info_sets_index_1, pure_strategies_2,info_sets_index_2)
            if state.is_chance_node:
                prob = state.chance_strategy[action]
                current_node_utilities += prob * child_utilities
            else:
                if node.information_set.player_id == 0:
                    index = info_sets_index_1[node.information_set]
                    # probs will be 1 if the action is in pure strategy else 0
                    probs = pure_strategies_1[:, index] == action
                    # convert to column vector so each row of matrix is multiplied by corresponding prob
                    probs = probs[:, np.newaxis,np.newaxis]
                    current_node_utilities += probs * child_utilities
                else:
                    index = info_sets_index_2[node.information_set]
                    probs = pure_strategies_2[:, index] == action
                    # convert to row vector so each column of matrix is multiplied by corresponding prob
                    probs = probs[np.newaxis, :, np.newaxis]
                    current_node_utilities += probs * child_utilities
    return current_node_utilities

    

def _collect_pure_strategies(info_sets):
    info_sets_index = {info_set: idx for idx, info_set in enumerate(info_sets)}
    action_spaces = [info_set.actions for info_set in info_sets]
    return np.array(list(itertools.product(*action_spaces))),info_sets_index


def convert_to_sequence_form(game_tree, information_set_dict) -> tuple[np.ndarray, ...]:
    """Convert an extensive-form game into its sequence-form representation.

    The sequence-form representation consists of:
        - The sequence-form payoff matrices for both players
        - The realization-plan constraint matrices and vectors for both players

    Feel free to conceptually split this function into smaller functions that compute
        - the sequences for a player, e.g. `_collect_sequences`
        - the sequence-form payoff matrix of a player, e.g. `_compute_sequence_form_payoff_matrix`
        - the realization-plan constraints of a player, e.g. `_compute_realization_plan_constraints`

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the sequence-form payoff matrices and realization-plan constraints.
    """
    # 1. Collect sequences (tuples of (InfoSet, ActionIndex))
    # We maintain a map from Sequence -> Index for O(1) lookups
    p1_sequences,seq_to_idx_1 = _collect_sequences(information_set_dict, 0)
    p2_sequences,seq_to_idx_2 = _collect_sequences(information_set_dict, 1)
    
    # 2. Build Constraint Matrices and Vectors (E x = e, F y = f)
    E, e = _build_constraint_matrix(p1_sequences, seq_to_idx_1, information_set_dict, 0,game_tree)
    F, f = _build_constraint_matrix(p2_sequences, seq_to_idx_2, information_set_dict, 1,game_tree)

    # 3. Build Payoff Matrices
    payoff_1 = np.zeros((len(p1_sequences), len(p2_sequences)))
    payoff_2 = np.zeros((len(p1_sequences), len(p2_sequences)))
    _fill_payoff_matrices(game_tree, p1_sequences, p2_sequences, payoff_1, payoff_2,seq_to_idx_1,seq_to_idx_2,information_set_dict)

    return payoff_1, payoff_2, E, e, F, f

def nodes_grouped_by_infoset(info_set_dict, game_tree):
    from collections import deque
    # Pre-create keys from info_set_dict values (keys in info_set_dict don't matter)
    # Pre-seed with all known information sets (optional but usually nice)
    # Pre-seed with all known information sets (optional but usually nice)
    out = {info_set: [] for info_set in info_set_dict.values()}

    q = deque([game_tree])
    while q:
        node = q.popleft()

        if node.information_set is not None:
            out[node.information_set].append(node)

        q.extend(node.children.values())

    return out

def find_children(obj, player_id,nodes_grouped_by_infoset_map,children):
    # 1. Determine which nodes to traverse from
    nodes_to_traverse = []
    if isinstance(obj, week07.information_set):
        nodes_to_traverse = nodes_grouped_by_infoset_map[obj]
    elif isinstance(obj, week07.nodes):
        nodes_to_traverse = [obj]
    else:
        nodes_to_traverse = obj # it's already a list of nodes
    
    # 2. Iterate through specific nodes and their children
    for node in nodes_to_traverse:
        for action_from_parent, child_node in node.children.items():
            
            # Case A: Found a node belonging to the target player
            if child_node.player_id == player_id:
                child_info_set = child_node.information_set
                
                # If valid info set, add a tuple for EACH legal action in that set
                if child_info_set is not None:
                    children.add(child_info_set)
            
            # Case B: Chance node (-1) or Opponent node -> Skip and Recurse
            else:
                # We pass the specific child_node to follow the path
                find_children(child_node, player_id,nodes_grouped_by_infoset_map, children)

    return children

def children_via_action(nodes,action):
    children = []
    for node in nodes:
        if action in node.children:
            child_node = node.children[action]
            children.append(child_node)
    return children

def find_all_children(all_info_sets,game_tree,nodes_grouped_by_infoset_map, player_id):
    parent_child_map = {}
    none_obj = False
    for obj in all_info_sets:
        none_obj = False
        if obj is None:
            obj = game_tree
            none_obj = True
        nodes_to_traverse = []
        if isinstance(obj, week07.information_set):
            nodes_to_traverse = nodes_grouped_by_infoset_map[obj]
            actions = obj.actions
        elif isinstance(obj, week07.nodes):
            nodes_to_traverse = [obj]
            actions = obj.children.keys()
        
        for action in actions:
            children = set()
            # find all children through specific action
            specific_nodes = children_via_action(nodes_to_traverse,action)
            children = find_children(specific_nodes, player_id,nodes_grouped_by_infoset_map, children)
            if children:
                #we found all children and now we turn them into sequences
                parent_child_map[(obj if not none_obj else None, action)] = [
                    (child, a)
                    for child in children
                    for a in child.actions
                ]
                #parent_child_map[(obj,action)] = children#[(children,a) for a in children.actions]

    return parent_child_map
      

        
def _build_constraint_matrix(sequences,seq_to_idx, info_set_dict, player_id,game_tree) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs the constraint matrix A and vector b such that A * r = b.
    Rows = Information Sets + 1 (Root)
    Cols = Sequences
    """
    # Map sequences to their column indices
    
    
    # Filter information sets belonging to this player
    player_info_sets = [None]
    player_info_sets.extend([iso for iso in info_set_dict.values() if iso.player_id == player_id])
    
    # Map InfoSets to Matrix Rows
    # Row 0 is the "Root" constraint (Empty Sequence prob = 1)
    # Rows 1..N are the conservation constraints for each InfoSet
    
    num_rows = len(player_info_sets)
    num_cols = len(sequences)
    
    mat = np.zeros((num_rows, num_cols))
    vec = np.zeros(num_rows)
    
    # The empty sequence (None) must have probability 1.
    mat[0, 0] = 1
    vec[0] = 1
    
    # --- Constraint 2: Information Set Flow (Parent - Children = 0) ---
    nodes_grouped_by_infoset_map = nodes_grouped_by_infoset(info_set_dict, game_tree)
    parent_child_map = find_all_children(player_info_sets,game_tree,nodes_grouped_by_infoset_map, player_id)
    i = 1
    for parent_seq, child_sequences in parent_child_map.items():
        if parent_seq[0] is None:
            parent_idx = 0
        else:
            parent_idx = seq_to_idx[parent_seq] 
        mat[i,parent_idx] = -1
        for child_seq in child_sequences:
            child_idx = seq_to_idx[child_seq]
            mat[i,child_idx] = 1
        i += 1
    return mat, vec

def find_nash_equilibrium_sequence_form(game_tree, information_set_dict) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function is expected to received an extensive-form game as input
    and convert it to its sequence-form using `convert_to_sequence_form`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of realization plans for the two players for representing a Nash equilibrium.
    """
    from scipy.optimize import linprog
    # 1. Get Sequence Form Matrices
    # A is payoff for P1. In zero-sum, payoff for P2 is -A.
    # E, e are constraints for P1: E @ x = e
    # F, f are constraints for P2: F @ y = f
    A, B, E, e, F, f = convert_to_sequence_form(game_tree, information_set_dict)
    
    # Dimensions
    num_seq_1 = A.shape[0] # Size of x
    num_seq_2 = A.shape[1] # Size of y
    num_con_1 = E.shape[0] # Size of e (constraints for P1)
    num_con_2 = F.shape[0] # Size of f (constraints for P2)

    # --- LP 1: Find Player 1's optimal realization plan (x) ---
    # We want to Maximize (f.T @ w) -> Minimize (-f.T @ w)
    # Variables vector z = [x, w]
    # x: realization plan for P1 (size: num_seq_1)
    # w: dual variables for P2's constraints (size: num_con_2)
    #
    # Constraints:
    # 1. E @ x = e                  (P1 sequence validity)
    # 2. -A.T @ x + F.T @ w <= 0    (P2 best response constraint)
    # 3. x >= 0, w is free
    
    c_p1 = np.concatenate([np.zeros(num_seq_1), -f])
    
    # Equality Matrix: [E, 0] * [x, w]^T = e
    A_eq_p1 = np.block([
        [E, np.zeros((num_con_1, num_con_2))]
    ])
    b_eq_p1 = e
    
    # Inequality Matrix: [-A.T, F.T] * [x, w]^T <= 0
    A_ub_p1 = np.block([
        [-A.T, F.T]
    ])
    b_ub_p1 = np.zeros(num_seq_2) # Corresponds to columns of A (P2 sequences)
    
    # Bounds: x >= 0, w is free (-inf, inf)
    bounds_p1 = [(0, None)] * num_seq_1 + [(None, None)] * num_con_2
    
    res_p1 = linprog(c_p1, A_ub=A_ub_p1, b_ub=b_ub_p1, A_eq=A_eq_p1, b_eq=b_eq_p1, bounds=bounds_p1, method='highs')
    
    if not res_p1.success:
        raise ValueError(f"LP for Player 1 failed: {res_p1.message}")
        
    x_optimal = res_p1.x[:num_seq_1]

    # --- LP 2: Find Player 2's optimal realization plan (y) ---
    # We want to Minimize (e.T @ u)
    # Variables vector z = [y, u]
    # y: realization plan for P2 (size: num_seq_2)
    # u: dual variables for P1's constraints (size: num_con_1)
    #
    # Constraints:
    # 1. F @ y = f                  (P2 sequence validity)
    # 2. A @ y - E.T @ u <= 0       (P1 best response constraint: E.T @ u >= A @ y)
    # 3. y >= 0, u is free

    c_p2 = np.concatenate([np.zeros(num_seq_2), e])

    # Equality Matrix: [F, 0] * [y, u]^T = f
    A_eq_p2 = np.block([
        [F, np.zeros((num_con_2, num_con_1))]
    ])
    b_eq_p2 = f

    # Inequality Matrix: [A, -E.T] * [y, u]^T <= 0
    A_ub_p2 = np.block([
        [A, -E.T]
    ])
    b_ub_p2 = np.zeros(num_seq_1) # Corresponds to rows of A (P1 sequences)

    # Bounds: y >= 0, u is free (-inf, inf)
    bounds_p2 = [(0, None)] * num_seq_2 + [(None, None)] * num_con_1

    res_p2 = linprog(c_p2, A_ub=A_ub_p2, b_ub=b_ub_p2, A_eq=A_eq_p2, b_eq=b_eq_p2, bounds=bounds_p2, method='highs')

    if not res_p2.success:
        raise ValueError(f"LP for Player 2 failed: {res_p2.message}")

    y_optimal = res_p2.x[:num_seq_2]

    # Clean up small numerical errors (negative zeros)
    x_optimal[x_optimal < 0] = 0
    y_optimal[y_optimal < 0] = 0

    return x_optimal, y_optimal




def convert_realization_plan_to_behavioural_strategy(realization_plan, sequences, info_set_dict):
    """
    Convert a realization plan vector into a behavioural strategy dictionary.
    Returns: Dict { InformationSet: np.array([prob_action_0, prob_action_1, ...]) }
    """
    behavioral_strategy = {}
    
    # Map sequences to their index for easy lookup
    seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
    
    # Group sequences by InformationSet
    # The sequences list contains tuples: (InfoSet, ActionIndex)
    # sequences[0] is None
    
    # Identify parent sequences (prob of reaching the info set)
    # We can reuse the logic: Parent Prob = Sum of Children Probs? No.
    # Parent Prob = Realization Plan value of the Parent Sequence.
    
    # Let's iterate over all Information Sets found in the sequence list
    # (Extract unique info sets from the keys in sequences)
    unique_info_sets = {s[0] for s in sequences if s is not None}
    
    for info_set in unique_info_sets:
        # 1. Find the parent sequence for this info set
        # We need the parent map logic again, or we can infer it:
        # The sum of realization plans for all actions at this info set 
        # SHOULD equal the probability of reaching this info set.
        
        child_probs = []
        child_seq_indices = []
        
        for i in range(len(info_set.actions)):
            seq_key = (info_set, i)
            idx = seq_to_idx[seq_key]
            prob = realization_plan[idx]
            child_probs.append(prob)
            child_seq_indices.append(idx)
            
        child_probs = np.array(child_probs)
        prob_reaching_infoset = np.sum(child_probs)
        
        if prob_reaching_infoset > 1e-9:
            # Normal case: normalize children by the sum
            strategy = child_probs / prob_reaching_infoset
        else:
            # If probability of reaching this node is 0, strategy can be arbitrary.
            # Uniform is a safe default.
            strategy = np.ones(len(info_set.actions)) / len(info_set.actions)
            
        behavioral_strategy[info_set] = strategy
        
    return behavioral_strategy


def _fill_payoff_matrices(game_tree, sequences_1, sequences_2, payoff_1, payoff_2, seq_to_idx_1, seq_to_idx_2, information_set_dict):
    
    # Dictionary to store utilities for compatible sequence pairs
    # Key: (p1_sequence_index, p2_sequence_index)
    # Value: (utility_p1, utility_p2)
    leaf_utilities = {}

    def traverse_and_map_utilities(node, p1_seq_idx, p2_seq_idx, chance_prob):
        """
        Recursively traverses the tree. When a leaf is reached, it records
        the utility for the specific pair of sequences (p1_idx, p2_idx) that led there.
        """
        state = node.state
        
        # --- Base Case: Leaf Node ---
        if state.terminated or state.truncated:
            u1, u2 = state.rewards
            
            # If this pair of sequences has been reached before (via a different chance branch),
            # we accumulate the weighted utility.
            key = (p1_seq_idx, p2_seq_idx)
            if key not in leaf_utilities:
                leaf_utilities[key] = np.zeros(2)
            
            leaf_utilities[key] += np.array([u1, u2]) * chance_prob
            return

        # --- Recursive Step: Chance Node ---
        if state.is_chance_node:
            for action, child in node.children.items():
                prob = state.chance_strategy[action]
                traverse_and_map_utilities(child, p1_seq_idx, p2_seq_idx, chance_prob * prob)

        # --- Recursive Step: Player Node ---
        else:
            player = state.current_player
            info_set = node.information_set
            
            for action, child in node.children.items():
                # Find the index of the new sequence created by this action
                action_idx_in_infoset = info_set.actions.index(action)
                new_seq_key = (info_set, action_idx_in_infoset)
                
                if player == 0: # Player 1 moves
                    new_idx = seq_to_idx_1[new_seq_key]
                    traverse_and_map_utilities(child, new_idx, p2_seq_idx, chance_prob)
                else: # Player 2 moves
                    new_idx = seq_to_idx_2[new_seq_key]
                    traverse_and_map_utilities(child, p1_seq_idx, new_idx, chance_prob)

    # 1. Pre-compute the compatibility map
    # We start with index 0 for both (The Empty Sequence)
    traverse_and_map_utilities(game_tree, 0, 0, 1.0)

    # 2. The Inner Function you requested
    def get_utility_if_compatible(idx_1, idx_2):
        """
        Checks if two sequence indices are compatible (i.e., they lead to a leaf together).
        Returns (u1, u2) if compatible, otherwise None.
        """
        return leaf_utilities.get((idx_1, idx_2), None)

    # 3. The Double Loop
    # We iterate through every possible pair of sequences
    for i in range(len(sequences_1)):
        for j in range(len(sequences_2)):
            
            result = get_utility_if_compatible(i, j)
            
            if result is not None:
                payoff_1[i, j] = result[0]
                payoff_2[i, j] = result[1]
            else:
                # Not compatible (cannot happen in the same history)
                # Payoff remains 0 (sparse matrix)
                pass
    


def _collect_sequences(information_set_dict, player_id):
    """
    Identifies all sequences for a player.
    """
    information_sets = [info_set for info_set in information_set_dict.values() if info_set.player_id == player_id]
    sequences = [None]  # Start with the empty sequence
    for info_set in information_sets:
        for action in info_set.actions:
            sequences.append((info_set, action))
    seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}
    return sequences, seq_to_idx



def convert_realization_plan_to_behavioural_strategy(*args, **kwargs):
    """Convert a realization plan to a behavioural strategy."""

    raise NotImplementedError


def main() -> None:
    from kuhn_poker import KuhnPokerNumpy as KuhnPoker
    env = KuhnPoker()

    # Initialize the environment with a random seed
    state = env.init(0)
    information_set_dict = {}
    game_tree = week07.traverse_tree(env, state, 2, information_set_dict,[],True)
    convert_to_sequence_form(game_tree, information_set_dict)
    find_nash_equilibrium_sequence_form(game_tree, information_set_dict)
    #matrix = convert_to_normal_form(game_tree, information_set_dict)
    #print(matrix)
    pass


if __name__ == '__main__':
    main()

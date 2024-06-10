# Settlers of Catan 
# Monte Carlo Tree Search class for AI players

from board import *
import simulation 
from player import *
from heuristicAIPlayer import *
import math
import copy

class Node:
    def __init__(self, state, action):
        self.parent = None

        #child nodes
        self.children = []

        # current state of game
        # state is dict {'board' = board, 'current_player' = ai_player, 'queue' = playerQueue}
        self.gameState = state 

        # total simulation rewards
        self.value = 0

        # visit count
        self.visits = 0

        # action reward
        self.action_reward = 0

        # action tuple
        self.action = action


class MCTS:
    """
    Monte-Carlo Tree Search. Create tree then choose best move(s) for current turn
    """
    def __init__(self, state, ppo_model, exploration_param=1):
        self.root_node = Node(state=state, action=()) # state is dict {'board' = board, 'current_player' = ai_player, 'queue' = playerQueue}
        self.exploration_param = exploration_param
        self.ppo_model = ppo_model

    def calcUCT(self, node):
        """
        Calculate UCT (Upper Confidence Bound) for a given node.
        """
        if node.visits == 0:
            return float('inf')

        # adjust exploration parameter based on how close the player is to winning
        explor_param = self.exploration_param
        if node.gameState['current_player'].victoryPoints >= 6:
            explor_param *= 0.5

        # weights for intermediate rewards and victory points
        intermediate_reward_weight = 0.5
        victory_point_weight = 1.0

        # increase weight of victory points dynamically as victory points approach 10
        if node.gameState['current_player'].victoryPoints >= 6:
            victory_point_weight = 2.0
        elif node.gameState['current_player'].victoryPoints >= 9:
            victory_point_weight = 5.0

        weighted_value = (intermediate_reward_weight * node.action_reward +
                        victory_point_weight * node.gameState['current_player'].victoryPoints)

        uct_value = weighted_value + node.value/ node.visits + explor_param * math.sqrt(math.log(node.parent.visits) / node.visits)

        # debug print
        #print(f"Node action: {node.action}, Action reward: {node.action_reward}, UCT value: {uct_value}")

        return uct_value
    
    def selection(self):
        """
        Select node with highest UCB for expansion
        """
        node = self.root_node
        while node.children:
            best_value = float('-inf')
            best_child = None
            for child in node.children:
                ucb_val = self.calcUCT(child)
                if ucb_val > best_value:
                    best_value = ucb_val
                    best_child = child
            node = best_child
        return node
            
    # helper to generate all legal actions from a state
    # every legal action contains end_turn
    def get_legal_actions(self, state):

        board = state['board']
        player = state['current_player']

        actions = []

        # Get potential actions just based on state (not including resources)
        potential_roads = board.get_potential_roads(player)
        potential_settlements = board.get_potential_settlements(player)
        potential_cities = board.get_potential_cities(player)

        # get resource counts
        num_bricks = player.resources['BRICK']
        num_wood = player.resources['WOOD']
        num_sheep = player.resources['SHEEP']
        num_wheat = player.resources['WHEAT']
        num_ore = player.resources['ORE']

        # Append to actions[]
        # Add road actions
        # add statement to only do building actions when resources are high and close to winning
        if player.victoryPoints >= 8 and all(value > 3 for value in player.resources.values()):
            # Add settlement building actions
            if num_bricks >= 1 and num_wood >= 1 and num_sheep >= 1 and num_wheat >= 1 and player.settlementsLeft > 0:
                for settlement in potential_settlements.keys():
                    #print(f'Possible settlement at {settlement}')
                    actions.append(('build_settlement', settlement))

            # Add city building actions
            if num_ore >= 3 and num_wheat >= 2 and player.citiesLeft > 0:
                for city in potential_cities.keys():
                    #print(f'Possible city at {city}')
                    actions.append(('build_city', city))
        else:
            # Add road actions
            for road, length in potential_roads.items():
                if num_bricks >= length and num_wood >= length and player.roadsLeft > 0:
                    #print('Possible road of length ' + str(length) + ' at ' + str(road[0]) + ' to ' + str(road[1]))
                    actions.append(('build_road', road[0], road[1], length))

            # Add settlement building actions
            if num_bricks >= 1 and num_wood >= 1 and num_sheep >= 1 and num_wheat >= 1 and player.settlementsLeft > 0:
                for settlement in potential_settlements.keys():
                    #print(f'Possible settlement at {settlement}')
                    actions.append(('build_settlement', settlement))

            # Add city building actions
            if num_ore >= 3 and num_wheat >= 2 and player.citiesLeft > 0:
                for city in potential_cities.keys():
                    #print(f'Possible city at {city}')
                    actions.append(('build_city', city))

            # Draw Development Card
            if num_wheat >= 1 and num_ore >= 1 and num_sheep >= 1 and not all(value == 0 for value in board.devCardStack.values()):
                #print('Possible action: draw_devCard')
                actions.append(('draw_devCard',))
            # Resource types
            
            resource_types = ['BRICK', 'WOOD', 'SHEEP', 'WHEAT', 'ORE']

            # Add trading actions with the bank
            for resource_1, amount_1 in player.resources.items():
                # Trade with the bank 4:1
                if amount_1 >= 4:
                    for resource_2 in resource_types:
                        if resource_1 != resource_2:
                            actions.append(('trade_with_bank', resource_1, resource_2))

                # General Trading Post 3:1
                if ('3:1 PORT' in player.portList) and (amount_1 >= 3):
                    for resource_2 in resource_types:
                        if resource_1 != resource_2:
                            actions.append(('trade_with_bank_3:1', resource_1, resource_2))

                # Specific Trading Port 2:1
                specific_port = f"2:1 {resource_1}"
                if specific_port in player.portList and amount_1 >= 2:
                    for resource_2 in resource_types:
                        if resource_1 != resource_2:
                            actions.append(('trade_with_bank_2:1', resource_1, resource_2))

        # adding "end turn" action
        actions.append(('end_turn', ))
        return actions
        
    # helper to apply an action and return the new state
    def apply_action(self, node, action):
        # create copy of the state
        new_state = {
            'board': node.gameState['board'].custom_copy(),
            'current_player': copy.deepcopy(node.gameState['current_player']),
            'queue': copy.deepcopy(node.gameState['queue'])
        }

        board = new_state['board']
        player = new_state['current_player']

        # Apply an action
        # action is a tuple with ('action', info ....)
        action_type = action[0]
        #print(f"Applying action: {action}")
        #check which action to take
        reward = 0
        if action_type == 'build_road':
            _, v1, v2, length = action
            for _ in range(length):
                player.build_road(v1, v2, board, sim=True)
                v1, v2 = v2, v1 
                reward = 1
                
            
        elif action_type == 'build_settlement':
            _, v = action
            player.build_settlement(v, board, sim=True)
            reward = 6

            
        elif action_type == 'build_city':
            _, v = action
            player.build_city(v, board, sim=True)
            reward = 5


        elif action_type == 'draw_devCard':
            player.draw_devCard(board, sim=True)
            reward = 0
            

        elif action_type in ['trade_with_bank', 'trade_with_bank_3:1', 'trade_with_bank_2:1']:
            _, resource1, resource2 = action
            player.trade_with_bank(resource1, resource2, sim=True)
            reward = -2

        elif action_type == 'end_turn':
            pass

        # update player in new_state
        new_state['current_player'] = player
        node.action_reward = reward

        return new_state

    
    def is_terminal_state(self, state):
        board = state['board']
        current_player = state['current_player']

        # check if current player has won (10 or more points)
        if current_player.victoryPoints >= 10:
            return True

        # check if the current player can make a legal action
        legal_actions = self.get_legal_actions({'board': board, 'current_player': current_player})
        if legal_actions:
            return False

        return True

    def expansion(self, node):
        """
        Expand node with new child node
        """

        # check if node is a terminal state
        if self.is_terminal_state(node.gameState):
            return 

        # if node is not already expanded, expand it
        if not node.children:
            
            # get all legal actions from current node
            legal_actions = self.get_legal_actions(node.gameState)
            #print(f"Num legal actions: {len(legal_actions)}")  # DEBUG PRINT
            # create new child node for each legal action
            for action in legal_actions:
                new_state = self.apply_action(node, action)
                child_node = Node(new_state, action)
                child_node.parent = node
                child_node.action_reward = node.action_reward #include action reward 
                # add child node
                node.children.append(child_node)
                #print(f"Added child with action: {action}")  # DEBUG PRINT
    
    def simulation(self, node):
        """
        Simulate game for the given child node
        Result: Win(+1) and Lose(-1) 
        """
        # for debugging sim_print=False, for actual sim_print=True
        simulate = simulation.catanAISimGame(state=node.gameState, ppo_model=self.ppo_model, sim_print=True) 
        result = simulate.get_result()
        return result

    def backpropagation(self, node, result):
        """
        Send reward back to ancestors of leaf
        """
        while node.parent is not None:
            node.visits += 1
            node.value += result #+ node.action_reward #now includes action reward
            node = node.parent
        node.visits += 1 # increment visit for root node

    def bestMove(self, iterations=100):
        """
        Select next best move for current node and return best move node.
        Always contains 'end_turn' move

        Note: Try pruning branches that show no improvement. Can help reduce runtime
            Additionally try simulating for one child node at a time?
        """
        for i in range(iterations):
            # select node with best UCT
            best_node = self.selection()
            # expand node
            self.expansion(best_node)
            # add a check to see if best_node has only 1 legal action ("end_turn"), not worth time to simulate
            if i==0 and len(best_node.children) == 1 and best_node.children[0].action[0] == 'end_turn':
                break
            # case where node is terminal
            #if best_node is None: 
            #    return ('end_turn',)
            for child in best_node.children: 
                #simulate
                result = self.simulation(child)
                #backpropagate
                self.backpropagation(child, result)

            #result = self.simulation(best_node.children[])
            # backpropagate
            #self.backpropagation(best_node, result)
        
        #print(f"finished iterations: {iterations}")
        if not self.root_node.children:
            return ('end_turn,')
        # find child with most visits
        max_visits = max([child.visits for child in self.root_node.children]) 
        for child in self.root_node.children:
            if child.visits == max_visits:
                return child.action
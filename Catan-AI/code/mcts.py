# Settlers of Catan 
# Monte Carlo Tree Search class for AI players

from board import *
from simulation import catanAIGame
from player import *
from heuristicAIPlayer import *
import math
import random
from collections import defaultdict

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

        self.uct = 0

        # action tuple
        self.action = action


class MCTS:
    """
    Monte-Carlo Tree Search. Create tree then choose best move(s) for current turn
    """
    def __init__(self, state, exploration_param=1):
        self.root_node = Node(state=state, action=()) # state is dict {'board' = board, 'current_player' = ai_player, 'queue' = playerQueue}
        self.exploration_param = exploration_param
        

    def calcUCT(self, node):
        """
        Calculate UST (Upper Confidence Bound) for given node
        """
        if node.visits == 0:
            return float('inf')

        return node.value/node.visits + self.exploration_param*math.sqrt(math.log(node.parent.visits) / node.visits)

    def selection(self, node):
        """
        Select node with highest UCB for expansion
        """
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
    def get_legal_actions(self, state):

        board = state['board']
        player = state['current_player']

        actions = []

        # Get potential actions just based on state (not including resources)
        potential_roads = board.get_potential_roads(player)
        potential_settlements  = board.get_potential_settlements(player)
        potential_cities = board.get_potential_cities(player)

        # get resource counts
        num_bricks = player.resources['BRICK']
        num_wood = player.resources['WOOD']
        num_sheep = player.resources['SHEEP']
        num_wheat = player.resources['WHEAT']
        num_ore = player.resources['ORE']

        # Append to actions[]
        # Add road actions
        if num_bricks >= 1 and num_wood >= 1:
            for road in potential_roads.keys():
                actions.append(('build_road', road[0], road[1]))

        # Add settlement building actions
        if num_bricks >= 1 and num_wood >= 1 and num_sheep >= 1 and num_wheat >= 1:
            for settlement in potential_settlements.keys():
                actions.append(('build_settlement', settlement))

        # Add city building actions
        if num_ore >= 3 and num_wheat >= 2:
            for city in potential_cities.keys():
                actions.append(('build_city', city))

        # Draw Development Card
        if num_wheat >= 1 and num_ore >= 1 and num_sheep >= 1:
            actions(('draw_devCard', ))

        # # Play Development Card
        # for dev_card, amount in player.devCards.items():
        #     if dev_card != 'VP' and amount > 0:
        #         actions.append(('play_devCard', dev_card))

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

            # Sepcific Trading Port 2:1
            specific_port = f"2:1 {resource_1}"
            if specific_port in player.portList and amount_1 >= 2:
                for resource_2 in resource_types:
                    if resource_1 != resource_2:
                        actions.append(('trade_with_bank_2:1', resource_1, resource_2))

        return actions
        
    # helper to apply an action and return the new state
    def apply_action(self, state, action):
        # create copy of the state
        new_state = state.copy()
        board = new_state['board']
        player = new_state['current_player']

        # Apply an action
        # action is a tuple with ('action', info ....)
        action_type = action[0]
        
        #check which action to take
        if action_type == 'build_road':
            _, v1, v2 = action
            player.build_road(v1, v2, board)
            
        elif action_type == 'build_settlement':
            _, v = action
            player.build_settlement(v, board)
            
        elif action_type == 'build_city':
            _, v = action
            player.build_city(v, board)

        elif action_type == 'draw_devCard':
            player.draw_devCard(board)
            
        # We aren't implementing playing a dev card
        # elif action_type == 'play_devCard':
        #     _, dev_card = action
        #     player.play_devCard()

        elif action_type == 'trade_with_bank' or action_type == 'trade_with_bank_3:1' or action_type == 'trade_with_bank_2:1':
            _, resource1, resource2 = action
            player.trade_with_bank(resource1, resource2)

        # update player in new_state
        new_state['current_player'] = player

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

            # create new child node for each legal action
            for action in legal_actions:
                new_state = self.apply_action(node.gameState, action)
                child_node = Node(new_state, action)
                child_node.parent = node
                # add child node
                node.children.append(child_node)
    

    def simulation(self, node):
        """
        Simulate game for the given child node
        Result: Win(+1) and Lose(-1) 
        """
        result = catanAIGame(node.gameState)
        return result

    def backpropagation(self, node, result):
        """
        Send reward back to ancestors of leaf
        """
        # call calcUST for node
        while node.parent is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def bestMove(self, iterations=200):
        """
        Select next best move for current node and return best move node
        """
        for i in range(iterations):
            # select node with best UCT
            best_node = self.selection(self.root_node)
            # expand node
            self.expansion(best_node)
            # simulate
            result = self.simulation(best_node)
            # backpropagate
            self.backpropagation(best_node, result)

        # find child with most visits
        max_visits = max(child.visits for child in self.root_node.children)
        for child in self.root_node.children:
            if child.visits == max_visits:
                return child.action
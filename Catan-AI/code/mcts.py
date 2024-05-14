# Settlers of Catan 
# Monte Carlo Tree Search class for AI players

from board import *
from catanGame import *
from player import *
from heuristicAIPlayer import *
import math
import random
from collections import defaultdict

class Node:
    def __init__(self, state, action):
        self.parent = None
        self.children = []
        self.gameState = state
        self.value = 0
        self.visits = 0
        self.uct = 0


class MCTS:
    """
    Monte-Carlo Tree Search. Create tree then choose best move(s) for current turn
    """
    def __init__(self, node, exploration_param=1):
        self.current_node = node
        self.exploration_param = exploration_param
        self.Q = defaultdict(int) # total simulation reward for each node
        self.N = defaultdict(int) # total number of visits
        

    def calcUCT(self, node):
        """
        Calculate UST (Upper Confidence Bound) for given node
        """
        return self.Q[node]/self.N[node] + self.exploration_param*math.sqrt(math.log(self.N[node]) / self.N[node])

    def selection(self, node):
        """
        Select node with highest UCB for expansion
        """
        pass

    def expansion(self, node):
        """
        Expand node with new child node
        """
        pass

    def simulation(self, node):
        """
        Simulate game for the given child node
        """
        pass

    def backpropagation(self, node):
        """
        Send reward back to ancestors of leaf
        """
        pass

    def bestMove(self, node):
        """
        Select next best move for current node
        Note: Best move may be a list
        """
        pass
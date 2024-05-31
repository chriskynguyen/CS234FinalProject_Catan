import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import random
import pygame

# imports for gym environment
import gym
from gym import spaces
import time

# imports for Catan game implementations
from code.board import *
from code.player import *
from code.heuristicAIPlayer import *

PLAYER_COLORS = ['black', 'darkslateblue', 'magenta4', 'orange1']


#function to check if a player has the longest road - after building latest road
def check_longest_road(self, player_i):
    if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
        longestRoad = True
        for p in list(self.playerQueue):
            if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                longestRoad = False
        
        if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
            #Set previous players flag to false and give player_i the longest road points
            for p in list(self.playerQueue):
                if(p.longestRoadFlag):
                    p.longestRoadFlag = False
                    p.victoryPoints -= 2

            player_i.longestRoadFlag = True
            player_i.victoryPoints += 2


class CatanEnv(gym.Env):
    #metadata = {"render_modes": []}

    def __init__(self, exploration_param):
        super(CatanEnv, self).__init__()
        self.board = None
        self.exploration_param = exploration_param
        self.num_players = 3
        self.players = []
        for player_id in range(1, self.num_players + 1):
            playerName = input("Enter Player {} name: ".format(player_id))
            newPlayer = heuristicAIPlayer(playerName, playerColor=PLAYER_COLORS[i-1], exploration_param=self.exploration_param)
            newPlayer.player_id = player_id
            self.players.append(newPlayer)
        self.current_player_idx = 0
        self.current_player = self.players[self.current_player_idx]
        self.turn_number = 0

        self.num_hex_tiles = 19
        self.num_vertices = 54
        self.num_edges = 72

        # Observation space for hex tiles: (resource type, dice number, robber)
        self.hex_tiles_space = spaces.Box(low=np.tile(np.array([0,2,0]),(self.num_hex_tiles, 1)), high=np.tile(np.array([4,12,1]), (self.num_hex_tiles, 1)), shape=(self.num_hex_tiles, 3), dtype=np.int32)
        
        # Observation space for vertices: (owner, building type, port, isColonised)
        self.vertices_space = spaces.Box(low=np.tile(np.array([0,0,0,0]),(self.num_vertices, 1)), high=np.tile(np.array([self.num_players,2,1,1]), (self.num_vertices, 1)), shape=(self.num_vertices, 4), dtype=np.int32)
        
        # Observation space for edges: (owner, is_road_built)
        self.edges_space = spaces.Box(low=np.tile(np.array([0,0]),(self.num_edges, 1)), high=np.tile(np.array([self.num_players,1]),(self.num_edges,1)), shape=(self.num_edges, 2), dtype=np.int32)
        
        # Observation space for player states
        self.player_states_space = spaces.Dict({
            'resources': spaces.Box(low=0, high=20, shape=(5,), dtype=np.int32),  # 5 types of resources, high = 20 upper limit of max resources
            'roads_left': spaces.Discrete(15),
            'settlements_left': spaces.Discrete(5),
            'cities_left': spaces.Discrete(4),
            'dev_cards': spaces.Box(low=0, high=5, shape=(5,), dtype=np.int32),  # 5 types of dev cards
            'victory_points': spaces.Discrete(11),  # 0 to 10 victory points
        })
        
        # Observation space for the game state
        self.game_state_space = spaces.Dict({
            'turn_number': spaces.Discrete(100),
            'current_player': spaces.Discrete(self.num_players),
        })
        
        # Combined all observation spaces
        self.observation_space = spaces.Dict({
            'hex_tiles': self.hex_tiles_space,
            'vertices': self.vertices_space,
            'edges': self.edges_space,
            'player_states': spaces.Tuple([self.player_states_space] * self.num_players),
            'game_state': self.game_state_space,
        })
        
        # TODO: Define the action space 
        self.action_space = spaces.Discrete(8)

    def _get_obs(self):
        """
        Get the current observation of the game state.
        Used in both reset and step
        """
        # Get hex tiles state
        hex_tiles_state = np.zeros((self.num_hex_tiles, 3), dtype=np.int32)
        for hexIndex, hexTile in self.board.hexTileDict.items():
            hex_tiles_state[hexIndex] = [hexTile.resource.type, hexTile.resource.num, hexTile.robber]
        
        # Get vertices state
        vertices_state = np.zeros((self.num_vertices,4), dtype=np.int32)
        for pixelCoords, vertex in self.board.boardGraph.items():
            player_id = vertex.state['Player'].player_id if vertex.state['Player'] is not None else 0
            building_type = 1 if vertex.state['Settlement'] else (2 if vertex.state['City'] else 0)
            port = 1 if vertex.port else 0
            colonized = 1 if vertex.isColonised else 0
            vertices_state[vertex.vertexIndex] = [player_id, building_type, port, colonized]

        # Get edges state
        edges_state = np.zeros((self.num_edges, 2), dtype=np.int32)
        unique_edges = set()
        edge_index = 0
        for vertex in self.board.boardGraph.keys():
            for edge_info in vertex.edgeState:
                if edge_info[0] is not None:  # If there is a player who built the road
                    player_id = edge_info[0].player_id
                    is_built = 1 if edge_info[1] else 0

                    # Create a unique identifier for the edge
                    for adjacent_vertex in vertex.edgeList:
                        edge = tuple(sorted((vertex.vertexIndex, adjacent_vertex.vertexIndex)))
                        if edge not in unique_edges:
                            unique_edges.add(edge)
                            edges_state[edge_index] = [player_id, is_built]
                            edge_index += 1
                
        # Get player states
        player_states = []
        for player in self.players:
            player_state = {
                'resources': np.array(player.resources, dtype=np.int32),
                'dev_cards': np.array(player.devCards, dtype=np.int32),
                'victory_points': player.victoryPoints,
            }
            player_states.append(player_state)
        
        # Get game state
        game_state = {
            'turn_number': self.turn_number,
            'current_player': self.current_player_idx,
        }
        
        observation = {
            'hex_tiles': hex_tiles_state,
            'vertices': vertices_state,
            'edges': edges_state,
            'player_states': tuple(player_states),
            'game_state': game_state,
        }
        
        return observation

    def build_initial_settlements(self):

        #Build Settlements and roads of each player forwards
        for player_i in self.players: 
            player_i.initial_setup(self.board)


        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in self.players.reverse(): 
            player_i.initial_setup(self.board)

            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1

    def reset(self):
        """
        Reset environment to its initial state and return the initial observation
        Used to start a new episode when after current one has met the terminal condition
        """
        # Reset environment
        self.board = catanBoard(is_copy=True) # is_copy=True means don't do prints
        self.dice_rolled = False
        for player in self.players:
            player.player_reset()
        self.build_initial_settlements()
        self.current_player_idx = 0
        self.current_player = self.players[self.current_player_idx]
        self.turn_number = 0

        return _get_obs()

    def apply_action(self, player, action):
        board = self.board
        action_type = action[0] 

        if action_type == 'build_road':
            _, v1, v2, length = action
            for _ in range(length):
                player.build_road(v1, v2, board, sim=True)
                
            
        elif action_type == 'build_settlement':
            _, v = action
            player.build_settlement(v, board, sim=True)

            
        elif action_type == 'build_city':
            _, v = action
            player.build_city(v, board, sim=True)


        elif action_type == 'draw_devCard':
            player.draw_devCard(board, sim=True)
            

        elif action_type in ['trade_with_bank', 'trade_with_bank_3:1', 'trade_with_bank_2:1']:
            _, resource1, resource2 = action
            player.trade_with_bank(resource1, resource2, sim=True)

        elif action_type == 'end_turn':
            pass

    def is_done(self, player):
        """
        Check if the game is done (e.g., if any player has reached 10 victory points).
        """
        if player.victoryPoints >= 10:
            return True
        return False
    
    #Function to roll dice 
    def rollDice(self):
        dice_1 = np.random.randint(1,7)
        dice_2 = np.random.randint(1,7)
        diceRoll = dice_1 + dice_2

        return diceRoll

    #Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if(diceRoll != 7): #Collect resources if not a 7
            #First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            #print('Resources rolled this turn:', hexResourcesRolled)

            #Check for each player
            for player_i in self.players:
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
        
        else:
            currentPlayer.heuristic_move_robber(self.board)

    def step(self, action):
        """
        Step through player move given a single action
        """
        done = False
        if not self.dice_rolled:
            diceNum = self.rollDice()
            self.update_playerResources(diceNum, self.current_player)
            self.dice_rolled = True
        
        self.apply_action(self.current_player, action)

        done = self.is_done(self.current_player)
        if action[0] == 'end_turn':
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            self.current_player = self.players[self.current_player_idx]
            self.turn_number += 1            
            self.dice_rolled = False

        reward = 1 if done else 0 #TODO: make reward function better
        observation = self._get_obs() 

        return observation, reward, done


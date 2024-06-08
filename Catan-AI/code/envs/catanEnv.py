import numpy as np
import torch
import os
import random
import pygame
from collections import Counter

# imports for gym environment
import gymnasium as gym
from gymnasium import spaces

# imports for Catan game implementations
from board import *
from player import *
from heuristicAIPlayer import *
from gameView import *


PLAYER_COLORS = ['black', 'darkslateblue', 'magenta4', 'orange1']
PLAYER_RESOURCE_TYPES = ['ORE', 'BRICK', 'WHEAT', 'WOOD', 'SHEEP']
RESOURCE_DICT = {'DESERT':0, 'ORE':1, 'BRICK':2, 'WHEAT':3, 'WOOD':4, 'SHEEP':5}

ACTIONS_ARRAY = [
    *[("build_road", v1, v2, length) for v1 in range(54) for v2 in range(54) for length in range(1, 4)],
    *[("build_settlement", v1) for v1 in range(54)],
    *[("build_city", v1) for v1 in range(54)],
    ("draw_devCard",),
    *[
        ("trade_with_bank", resource_1, resource_2) 
        for resource_1 in PLAYER_RESOURCE_TYPES 
        for resource_2 in PLAYER_RESOURCE_TYPES 
        if resource_1 != resource_2
    ],
    *[
        ("trade_with_bank_3:1", resource_1, resource_2) 
        for resource_1 in PLAYER_RESOURCE_TYPES 
        for resource_2 in PLAYER_RESOURCE_TYPES 
        if resource_1 != resource_2
    ],
    *[
        ("trade_with_bank_2:1", resource_1, resource_2) 
        for resource_1 in PLAYER_RESOURCE_TYPES 
        for resource_2 in PLAYER_RESOURCE_TYPES 
        if resource_1 != resource_2
    ],
    ("end_turn",),
]
ACTIONS_LEN = len(ACTIONS_ARRAY)
PLAYER__NAMES = ["Active", "Fixed Policy 1", "Fixed Policy 2"]

class CatanEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self):
        super(CatanEnv, self).__init__()
        self.board = None
        self.dice_rolled = False
        self.fixed_policy = None

        self.num_players = 3
        self.players = []
        for i in range(1, self.num_players+1):
            newPlayer = heuristicAIPlayer(PLAYER__NAMES[i-1], playerColor=PLAYER_COLORS[i-1], exploration_param=0)
            newPlayer.player_id = i # 0 is reserved for no players, specifically in edge state
            newPlayer.updateAI()
            self.players.append(newPlayer)
        self.current_player_idx = 0
        self.current_player = self.players[self.current_player_idx]

        self.turn_number = 0
        self.max_episode_steps = 1000

        self.num_hex_tiles = 19
        self.num_vertices = 54
        self.num_edges = 72

        # Observation space for hex tiles: (resource type, dice number, robber)
        self.hex_tiles_space = spaces.Box(low=np.tile(np.array([0,1,0]),(self.num_hex_tiles, 1)), high=np.tile(np.array([5,12,1]), (self.num_hex_tiles, 1)), shape=(self.num_hex_tiles, 3), dtype=np.int32)
        
        # Observation space for vertices: (owner, building type, port, isColonised)
        self.vertices_space = spaces.Box(low=np.tile(np.array([0,0,0,0]),(self.num_vertices, 1)), high=np.tile(np.array([self.num_players,2,1,1]), (self.num_vertices, 1)), shape=(self.num_vertices, 4), dtype=np.int32)
        
        # Observation space for edges: (owner, is_road_built)
        self.edges_space = spaces.Box(low=np.tile(np.array([0,0]),(self.num_edges, 1)), high=np.tile(np.array([self.num_players,1]),(self.num_edges,1)), shape=(self.num_edges, 2), dtype=np.int32)

        # Observation space for player states
        self.player_states_spaces = spaces.Box(low=0, high=20, shape=(self.num_players * 14,), dtype=np.int32)
        
        # Observation space for the game state: (turn number, current player)
        self.game_state_space = spaces.Box(low=0, high=120, shape=(2,), dtype=np.int32)
        
        # Combined all observation spaces
        self.observation_space = spaces.Dict({
            'hex_tiles': self.hex_tiles_space,
            'vertices': self.vertices_space,
            'edges': self.edges_space,
            'player_states': self.player_states_spaces,
            'game_state': self.game_state_space,
        })
        
        # Define the action space 
        self.action_space = spaces.Discrete(ACTIONS_LEN)

        self.render_mode = "human"
        self.view = None
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Get the current observation of the game state.
        Used in both reset and step
        """
        # Get hex tiles state
        hex_tiles_state = []
        for hexIndex, hexTile in self.board.hexTileDict.items():
            resource_type = RESOURCE_DICT[hexTile.resource.type]
            robber = 1 if hexTile.robber else 0
            dice_num = hexTile.resource.num if hexTile.resource.num is not None else 1
            hex_tiles_state.append([resource_type, dice_num, hexTile.robber])
        hex_tiles_state = np.array(hex_tiles_state, dtype=np.int32)
        
        # Get vertices state
        vertices_state = []
        for pixelCoords, vertex in self.board.boardGraph.items():
            player_id = vertex.state['Player'].player_id if vertex.state['Player'] is not None else 0
            building_type = 1 if vertex.state['Settlement'] else (2 if vertex.state['City'] else 0)
            port = 1 if vertex.port else 0
            colonized = 1 if vertex.isColonised else 0
            vertices_state.append([player_id, building_type, port, colonized])
        vertices_state = np.array(vertices_state, dtype=np.int32)

        # Get edges state 
        edges_state = []
        unique_edges = set()
        edge_index = 0
        for vertex in self.board.boardGraph.values():
            for edge_info, adjacent_pixelCoord in zip(vertex.edgeState, vertex.edgeList):
                # Initialize the adjacent vertex
                adjacent_vertex = None
                for v in self.board.boardGraph.values():
                    if v.getVertex_fromPixel(adjacent_pixelCoord):
                        adjacent_vertex = v
                        break

                if adjacent_vertex is not None:
                    # Create a unique identifier for the edge
                    edge = tuple(sorted((vertex.vertexIndex, adjacent_vertex.vertexIndex)))
                    if edge not in unique_edges:
                        unique_edges.add(edge)
                        if edge_info[0] is not None:  # If there is a player who built the road
                            player_id = edge_info[0].player_id
                            is_built = 1 if edge_info[1] else 0
                        else:  # No player built the road
                            player_id = 0
                            is_built = 0

                        edges_state.append([player_id, is_built])

        edges_state = np.array(edges_state, dtype=np.int32)
                
        # Get player states
        player_states = []
        for player in self.players:
            resources = [num for num in player.resources.values()]
            dev_cards = [num for num in player.devCards.values()]
            player_state = resources + [player.roadsLeft, player.settlementsLeft, player.citiesLeft] + dev_cards + [player.victoryPoints]
            player_states.extend(player_state)
        player_states = np.array(player_states, dtype=np.int32)

        # Get game state
        game_state = np.array([self.turn_number, self.current_player_idx], dtype=np.int32)
        
        observation = {
            'hex_tiles': hex_tiles_state,
            'vertices': vertices_state,
            'edges': edges_state,
            'player_states': player_states,
            'game_state': game_state,
        }
        
        return observation

    def _get_info(self, end_turn):
        return {'turn_done': end_turn}

    def build_initial_settlements(self):

        playerList = list(self.players)
        #Build Settlements and roads of each player forwards
        for player_i in playerList: 
            player_i.initial_setup(self.board, sim=True)


        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList: 
            player_i.initial_setup(self.board, sim=True)

            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1

    def reset(self, seed=None, options=None):
        """
        Reset environment to its initial state and return the initial observation
        Used to start a new episode when after current one has met the terminal condition
        """
        super().reset(seed=seed)

        # Reset environment
        self.board = catanBoard(is_copy=True) # is_copy=True means don't do prints

        self.view = catanGameView(self.board, self) # Initialize the view
        if self.render_mode == "human":
            pygame.init()
            self.clock = pygame.time.Clock()

        self.dice_rolled = False
        for player in self.players:
            player.player_reset()
        self.build_initial_settlements()
        self.current_player_idx = 0
        self.current_player = self.players[self.current_player_idx]
        self.turn_number = 0

        return self._get_obs(), self._get_info(end_turn=False)

    def get_vertex_from_idx(self, v):
        for vertex in self.board.boardGraph.values():
            if vertex.vertexIndex == v:
                return vertex.pixelCoordinates

    def apply_action(self, player, action):
        board = self.board
        action = ACTIONS_ARRAY[action]
        action_type = action[0] 

        if action_type == 'build_road':
            _, v1, v2, length = action
            for _ in range(length):
                player.build_road(self.get_vertex_from_idx(v1), self.get_vertex_from_idx(v2), board, sim=True)
                
            
        elif action_type == 'build_settlement':
            _, v = action
            player.build_settlement(self.get_vertex_from_idx(v), board, sim=True)

            
        elif action_type == 'build_city':
            _, v = action
            player.build_city(self.get_vertex_from_idx(v), board, sim=True)


        elif action_type == 'draw_devCard':
            player.draw_devCard(board, sim=True)
            

        elif action_type in ['trade_with_bank', 'trade_with_bank_3:1', 'trade_with_bank_2:1']:
            _, resource1, resource2 = action
            player.trade_with_bank(resource1, resource2, sim=True)

        elif action_type == 'end_turn':
            pass

    def is_done(self, player):
        """
        Check if the game is done to terminate episode
        (e.g., if any player has reached 10 victory points).
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
            currentPlayer.heuristic_move_robber(self.board, sim=True)

    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in self.players:
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                for p in self.players:
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2

                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

    # Function for external use
    def update_fixed_policy(self, model):
        self.fixed_policy = model

    # every legal action contains end_turn
    def get_legal_actions(self):

        board = self.board
        player = self.current_player

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
                    actions.append(('build_settlement', self.board.boardGraph[settlement].vertexIndex))

            # Add city building actions
            if num_ore >= 3 and num_wheat >= 2 and player.citiesLeft > 0:
                for city in potential_cities.keys():
                    actions.append(('build_city', self.board.boardGraph[city].vertexIndex))
        else:
            # Add road actions
            for road, length in potential_roads.items():
                if num_bricks >= length and num_wood >= length and player.roadsLeft > 0:
                    actions.append(('build_road', self.board.boardGraph[road[0]].vertexIndex, self.board.boardGraph[road[1]].vertexIndex, length))

            # Add settlement building actions
            if num_bricks >= 1 and num_wood >= 1 and num_sheep >= 1 and num_wheat >= 1 and player.settlementsLeft > 0:
                for settlement in potential_settlements.keys():
                    actions.append(('build_settlement', self.board.boardGraph[settlement].vertexIndex))

            # Add city building actions
            if num_ore >= 3 and num_wheat >= 2 and player.citiesLeft > 0:
                for city in potential_cities.keys():
                    actions.append(('build_city', self.board.boardGraph[city].vertexIndex))

            # Draw Development Card
            if num_wheat >= 1 and num_ore >= 1 and num_sheep >= 1 and not all(value == 0 for value in board.devCardStack.values()):
                actions.append(('draw_devCard',))

            trading_actions = []
            # Add trading actions with the bank
            for resource_1, amount_1 in player.resources.items():
                # Trade with the bank 4:1
                if amount_1 >= 4:
                    for resource_2 in PLAYER_RESOURCE_TYPES:
                        if resource_1 != resource_2:
                            trading_actions.append(('trade_with_bank', resource_1, resource_2))

                # General Trading Post 3:1
                if ('3:1 PORT' in player.portList) and (amount_1 >= 3):
                    for resource_2 in PLAYER_RESOURCE_TYPES:
                        if resource_1 != resource_2:
                            trading_actions.append(('trade_with_bank_3:1', resource_1, resource_2))

                # Specific Trading Port 2:1
                specific_port = f"2:1 {resource_1}"
                if specific_port in player.portList and amount_1 >= 2:
                    for resource_2 in PLAYER_RESOURCE_TYPES:
                        if resource_1 != resource_2:
                            trading_actions.append(('trade_with_bank_2:1', resource_1, resource_2))

            actions_count = Counter(action for action, *_ in trading_actions)
            most_common_trades = [action for action, count in actions_count.items() if count == max(actions_count.values())]
            filtered_trades = [tup for tup in trading_actions if tup[0] in most_common_trades]
            actions.extend(filtered_trades)

        # adding "end turn" action
        actions.append(('end_turn', ))

        actions_array_np = np.array(ACTIONS_ARRAY, dtype=object)

        valid_actions = [np.where([np.array_equal(action, legal_action) for action in actions_array_np])[0] for legal_action in actions]
        return valid_actions


    def step(self, action):
        """
        Step through player move given a single action
        """
        terminated = False # game over, end episode
        end_turn = False
        truncated = False # truncated if passes max steps per episode

        if not self.dice_rolled:
            diceNum = self.rollDice()
            self.update_playerResources(diceNum, self.current_player)
            self.dice_rolled = True
        
        if self.current_player_idx != 0 and self.fixed_policy is not None:
            obs = self._get_obs()
            action, _states = self.fixed_policy.predict(obs)
        #print(ACTIONS_ARRAY[action]) #DEBUG
        self.apply_action(self.current_player, action)
        self.check_longest_road(self.current_player)

        if self.turn_number >= self.max_episode_steps:
            truncated = True
        terminated = self.is_done(self.current_player)

        if ACTIONS_ARRAY[action][0] == 'end_turn':
            end_turn = True
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            self.current_player = self.players[self.current_player_idx]
            self.turn_number += 1            
            self.dice_rolled = False

        reward = 1 if terminated and self.current_player_idx==0 else 0 #TODO: make reward function better
        observation = self._get_obs() 
        info = self._get_info(end_turn=end_turn)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.view is None:
            raise ValueError("View has not been initialized. Call reset() before rendering.")

        self.view.displayGameScreen()
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    

    
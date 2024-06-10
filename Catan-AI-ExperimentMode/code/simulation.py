#Settlers of Catan
#Gameplay class with pygame with AI players

from board import *
from gameView import *
from player import *
from heuristicAIPlayer import *
#import queue
from collections import deque
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt
from sb3_contrib.ppo_mask import MaskablePPO

RESOURCE_DICT = {'DESERT':0, 'ORE':1, 'BRICK':2, 'WHEAT':3, 'WOOD':4, 'SHEEP':5}
PLAYER_RESOURCE_TYPES = ['ORE', 'BRICK', 'WHEAT', 'WOOD', 'SHEEP']
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

#Class to implement an only AI
class catanAISimGame():
    #Create gameboard from current board in mcts.py
    def __init__(self, state, ppo_model, sim_print=True):
        self.board = state["board"]

        #Game State variables
        self.gameOver = False
        self.maxPoints = 10
        self.numTurns = 0
        self.player_name = state["current_player"].name
        self.result = -1
        #Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = state["queue"]

        #sim_print False means to print out information
        self.sim_print = sim_print

        self.ppo_model = ppo_model
        self.playSimCatan()
    
    def get_result(self):
        return self.result

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
            for player_i in list(self.playerQueue):
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            #print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            #print("{} collects 2 {} from City".format(player_i.name, resourceGenerated))

                #print("Player:{}, Resources:{}, Points: {}".format(player_i.name, player_i.resources, player_i.victoryPoints))
                #print('Dev Cards:{}'.format(player_i.devCards))
                #print("RoadsLeft:{}, SettlementsLeft:{}, CitiesLeft:{}".format(player_i.roadsLeft, player_i.settlementsLeft, player_i.citiesLeft))
                #print('MaxRoadLength:{}, Longest Road:{}\n'.format(player_i.maxRoadLength, player_i.longestRoadFlag))
        
        else:
            #print("AI using heuristic robber...")
            currentPlayer.heuristic_move_robber(self.board, self.sim_print)


    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in list(self.playerQueue):
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                prevPlayer = ''
                for p in list(self.playerQueue):
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                #print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

    #function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if(player_i.knightsPlayed >= 3): #Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue):
                if(p.knightsPlayed >= player_i.knightsPlayed and p != player_i): #Check if any other players have more knights played
                    largestArmy = False
            
            if(largestArmy and player_i.largestArmyFlag == False): #if player_i takes largest army and didn't already have it
                #Set previous players flag to false and give player_i the largest points
                prevPlayer = ''
                for p in list(self.playerQueue):
                    if(p.largestArmyFlag):
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                #print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))


    #Wrapper function to control all trading
    def trade(self, player_i):
        for r1, r1_amount in player_i.resources.items():
            if(r1_amount >= 6): #heuristic to trade if a player has more than 5 of a particular resource
                for r2, r2_amount in player_i.resources.items():
                    if(r2_amount < 1):
                        player_i.trade_with_bank(r1, r2, self.sim_print)
                        break

    def convert_to_observation(self, player_i):
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
        for player in list(self.playerQueue):
            resources = [num for num in player.resources.values()]
            dev_cards = [num for num in player.devCards.values()]
            player_state = resources + [player.roadsLeft, player.settlementsLeft, player.citiesLeft] + dev_cards + [player.victoryPoints]
            player_states.extend(player_state)
        player_states = np.array(player_states, dtype=np.int32)

        # Get game state
        game_state = np.array([self.numTurns, self.playerQueue.index(self.player_i)], dtype=np.int32)
        
        observation = {
            'hex_tiles': hex_tiles_state,
            'vertices': vertices_state,
            'edges': edges_state,
            'player_states': player_states,
            'game_state': game_state,
        }
        
        return observation

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
            return True
        return False

    # function to simulate moves
    # implement with PPO
    def sim_move(self, board, player_i):
        if player_i.usePPO:
            done = False
            while not done:
                obs = self.convert_to_observation(player_i)
                action, _ = self.ppo_model.predict(obs)
                done = self.apply_action(player_i, action)
        else:
            self.trade(player_i)
            #Build a settlements
            possibleVertices = board.get_potential_settlements(player_i)
            if(possibleVertices and player_i.resources['BRICK'] > 0 and player_i.resources['WOOD'] > 0 and player_i.resources['SHEEP'] > 0 and player_i.resources['WHEAT'] > 0):
                randomVertex = np.random.randint(0, len(possibleVertices.keys()))
                player_i.build_settlement(list(possibleVertices.keys())[randomVertex], board, self.sim_print)

            #Build a City
            possibleVertices = board.get_potential_cities(player_i)
            if(possibleVertices and player_i.resources['WHEAT'] >= 2 and player_i.resources['ORE'] >= 3):
                randomVertex = np.random.randint(0, len(possibleVertices.keys()))
                player_i.build_city(list(possibleVertices.keys())[randomVertex], board, self.sim_print)

            #Build a couple roads
            for i in range(2):
                if player_i.resources['BRICK'] > 0 and player_i.resources['WOOD'] > 0:
                    possibleRoads = board.get_potential_roads(player_i)
                    if possibleRoads: # add check to see if there are no available roads
                        randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                        player_i.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board, self.sim_print)

            #Draw a Dev Card with 1/3 probability
            devCardNum = np.random.randint(0, 3)
            if devCardNum == 0:
                player_i.draw_devCard(board, self.sim_print)

    #Function that runs the main game loop with all players and pieces
    def playSimCatan(self):
        #self.board.displayBoard() #Display updated board
        numTurns = 0
        while not self.gameOver:
            #Loop for each player's turn -> iterate through the player queue starting at next player
            for currPlayer in self.playerQueue:
                numTurns += 1
                #print("AI Player {} playing...".format(currPlayer.name)) #DEBUG
                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while not turnOver:
                    #Roll Dice and update player resources and dice stats
                    #pygame.event.pump()
                    # Don't roll dice when entering in sim (already did dice roll in actual gameplay)
                    diceNum = self.rollDice()
                    self.update_playerResources(diceNum, currPlayer)
                    diceRolled = True
                    
                    self.sim_move(self.board, currPlayer) #AI Player makes all its moves
                    #Check if AI player gets longest road and update Victory points
                    self.check_longest_road(currPlayer)
                    
                    #self.boardView.displayGameScreen()#Update back to original gamescreen
                    #pygame.time.delay(300)
                    turnOver = True
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        break

                if self.gameOver: #TODO: test numTurns > __ 
                    if currPlayer.name == self.player_name:
                        self.result = 1
                    #startTime = pygame.time.get_ticks()
                    #runTime = 0
                    #while(runTime < 5000): #5 second delay prior to quitting
                    #    runTime = pygame.time.get_ticks() - startTime

                    break
                                   
from player import *
import numpy as np
from mcts import *
import copy

# for loading ppo policy
from sb3_contrib.ppo_mask import MaskablePPO

class heuristicAIPlayer(player):
    def __init__(self, playerName, playerColor, usePPO=False, exploration_param=0.5, strategy="heuristic"):
        super().__init__(playerName, playerColor, usePPO, exploration_param)
        self.strategy = strategy

    def updateAI(self):
        self.isAI = True
        self.setupResources = []
        self.resources = {'ORE': 0, 'BRICK': 4, 'WHEAT': 2, 'WOOD': 4, 'SHEEP': 2}
        print("Added new AI Player:", self.name)

    def initial_setup(self, board):
        possibleVertices = board.get_setup_settlements(self)
        diceRoll_expectation = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1, None: 0}
        vertexValues = []

        for v in possibleVertices.keys():
            vertexNumValue = 0
            resourcesAtVertex = []
            for adjacentHex in board.boardGraph[v].adjacentHexList:
                resourceType = board.hexTileDict[adjacentHex].resource.type
                if resourceType not in resourcesAtVertex:
                    resourcesAtVertex.append(resourceType)
                numValue = board.hexTileDict[adjacentHex].resource.num
                vertexNumValue += diceRoll_expectation[numValue]

            vertexNumValue += len(resourcesAtVertex) * 2
            for r in resourcesAtVertex:
                if r != 'DESERT' and r not in self.setupResources:
                    vertexNumValue += 2.5

            vertexValues.append(vertexNumValue)

        vertexToBuild_index = vertexValues.index(max(vertexValues))
        vertexToBuild = list(possibleVertices.keys())[vertexToBuild_index]

        for adjacentHex in board.boardGraph[vertexToBuild].adjacentHexList:
            resourceType = board.hexTileDict[adjacentHex].resource.type
            if resourceType not in self.setupResources and resourceType != 'DESERT':
                self.setupResources.append(resourceType)

        self.build_settlement(vertexToBuild, board)

        possibleRoads = board.get_setup_roads(self)
        randomEdge = np.random.randint(0, len(possibleRoads.keys()))
        self.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)

    def run_action(self, action, board):
        action_type = action[0]
        if action_type == 'build_road':
            _, v1, v2, length = action
            self.build_road(v1, v2, board)
        elif action_type == 'build_settlement':
            _, v = action
            self.build_settlement(v, board)
        elif action_type == 'build_city':
            _, v = action
            self.build_city(v, board)
        elif action_type == 'draw_devCard':
            self.draw_devCard(board)
        elif action_type in ['trade_with_bank', 'trade_with_bank_3:1', 'trade_with_bank_2:1']:
            _, resource1, resource2 = action
            self.trade_with_bank(resource1, resource2)

    def move(self, board, queue):
        print(f"AI Player {self.name} playing...")
        for _ in range(10):
            if self.victoryPoints > 9:
                break
            state = {'board': board, 'current_player': self, 'queue': queue}
            model = MaskablePPO.load('./results/Catan-ppo/final_model.zip')

            tree = MCTS(state, model, self.exploration_param)
            action = tree.bestMove(iterations=1000) # we can increase this number to get better results
            if action[0] == 'end_turn':
                break
            self.run_action(action, board)
        return

    def trade(self):
        for r1, r1_amount in self.resources.items():
            if r1_amount >= 6:
                for r2, r2_amount in self.resources.items():
                    if r2_amount < 1:
                        self.trade_with_bank(r1, r2)
                        break

    def choose_player_to_rob(self, board):
        robberHexDict = board.get_robber_spots()
        hexToRob_index = None
        playerToRob_hex = None
        maxHexScore = float('-inf')

        for hex_ind, hexTile in robberHexDict.items():
            vertexList = polygon_corners(board.flat, hexTile.hex)
            hexScore = 0
            playerToRob_VP = 0
            playerToRob = None
            for vertex in vertexList:
                playerAtVertex = board.boardGraph[vertex].state['Player']
                if playerAtVertex == self:
                    hexScore -= self.victoryPoints
                elif playerAtVertex is not None:
                    hexScore += playerAtVertex.visibleVictoryPoints
                    if playerAtVertex.visibleVictoryPoints >= playerToRob_VP and sum(playerAtVertex.resources.values()) > 0:
                        playerToRob_VP = playerAtVertex.visibleVictoryPoints
                        playerToRob = playerAtVertex
                else:
                    pass

            if hexScore >= maxHexScore and playerToRob is not None:
                hexToRob_index = hex_ind
                playerToRob_hex = playerToRob
                maxHexScore = hexScore

        if hexToRob_index is None:
            hexToRob_index = list(robberHexDict.keys())[0]
            playerToRob_hex = None
        return hexToRob_index, playerToRob_hex

    def heuristic_move_robber(self, board, sim=False):
        hex_i, playerRobbed = self.choose_player_to_rob(board)
        self.move_robber(hex_i, board, playerRobbed, sim)
        return

    def heuristic_move(self, board):
        print(f"AI Player {self.name} playing...")
        self.trade()
        possibleVertices = board.get_potential_settlements(self)
        if possibleVertices != {} and (self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0 and self.resources['SHEEP'] > 0 and self.resources['WHEAT'] > 0):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_settlement(list(possibleVertices.keys())[randomVertex], board)

        possibleVertices = board.get_potential_cities(self)
        if possibleVertices != {} and (self.resources['WHEAT'] >= 2 and self.resources['ORE'] >= 3):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_city(list(possibleVertices.keys())[randomVertex], board)

        for i in range(2):
            if self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0:
                possibleRoads = board.get_potential_roads(self)
                randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                self.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)

        devCardNum = np.random.randint(0, 3)
        if devCardNum == 0:
            self.draw_devCard(board)

        return

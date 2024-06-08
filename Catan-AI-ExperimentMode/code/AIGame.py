import argparse
import json
from collections import deque
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt
import copy
from board import *
from gameView import *
from player import *
from heuristicAIPlayer import *

class catanAIGame():
    def __init__(self, numPlayers, playerConfigs):
        print("Initializing Settlers of Catan with only AI Players...")
        self.board = catanBoard(is_copy=False)

        self.gameOver = False
        self.maxPoints = 10
        self.numPlayers = numPlayers

        self.diceStats = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
        self.diceStats_list = []

        print(f"Initializing game with {self.numPlayers} players...")
        print("Note that Player 1 goes first, Player 2 second and so forth.")

        self.playerQueue = deque()
        self.gameSetup = True

        self.boardView = catanGameView(self.board, self)

        self.build_initial_settlements(playerConfigs)
        self.results = self.playCatan()

        plt.hist(self.diceStats_list, bins=11)
        #plt.show()

    def build_initial_settlements(self, playerConfigs):
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            config = playerConfigs[i]
            strategy = config.get("strategy", "heuristic")
            newPlayer = heuristicAIPlayer(config["name"], playerColors[i], config["usePPO"], config["exploration_param"], strategy)
            newPlayer.player_id = i + 1
            newPlayer.updateAI()
            self.playerQueue.append(newPlayer)

        playerList = list(self.playerQueue)

        for player_i in playerList:
            player_i.initial_setup(self.board)
            pygame.event.pump()
            self.boardView.displayGameScreen()
            #pygame.time.delay(1000) # no delay during experiments

        playerList.reverse()
        for player_i in playerList:
            player_i.initial_setup(self.board)
            pygame.event.pump()
            self.boardView.displayGameScreen()
            #pygame.time.delay(1000) # no delay during experiments

            print(f"Player {player_i.name} starts with {len(player_i.setupResources)} resources")

            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if resourceGenerated != 'DESERT':
                    player_i.resources[resourceGenerated] += 1
                    print(f"{player_i.name} collects 1 {resourceGenerated} from Settlement")

        #pygame.time.delay(5000) # no delay during experiments
        self.gameSetup = False

    def rollDice(self):
        dice_1 = np.random.randint(1, 7)
        dice_2 = np.random.randint(1, 7)
        diceRoll = dice_1 + dice_2
        print("Dice Roll = ", diceRoll, "{", dice_1, dice_2, "}")

        return diceRoll

    def update_playerResources(self, diceRoll, currentPlayer):
        if diceRoll != 7:
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)

            for player_i in list(self.playerQueue):
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList:
                        if adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False:
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            print(f"{player_i.name} collects 1 {resourceGenerated} from Settlement")

                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList:
                        if adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False:
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            print(f"{player_i.name} collects 2 {resourceGenerated} from City")

                print(f"Player:{player_i.name}, Resources:{player_i.resources}, Points: {player_i.victoryPoints}")
                print(f'MaxRoadLength:{player_i.maxRoadLength}, Longest Road:{player_i.longestRoadFlag}\n')

        else:
            print("AI using heuristic robber...")
            currentPlayer.heuristic_move_robber(self.board)

    def check_longest_road(self, player_i):
        if player_i.maxRoadLength >= 5:
            longestRoad = True
            for p in list(self.playerQueue):
                if p.maxRoadLength >= player_i.maxRoadLength and p != player_i:
                    longestRoad = False

            if longestRoad and player_i.longestRoadFlag == False:
                prevPlayer = ''
                for p in list(self.playerQueue):
                    if p.longestRoadFlag:
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name

                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                print(f"Player {player_i.name} takes Longest Road {prevPlayer}")

    def check_largest_army(self, player_i):
        if player_i.knightsPlayed >= 3:
            largestArmy = True
            for p in list(self.playerQueue):
                if p.knightsPlayed >= player_i.knightsPlayed and p != player_i:
                    largestArmy = False

            if largestArmy and player_i.largestArmyFlag == False:
                prevPlayer = ''
                for p in list(self.playerQueue):
                    if p.largestArmyFlag:
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name

                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                print(f"Player {player_i.name} takes Largest Army {prevPlayer}")

    def playCatan(self):
        numTurns = 0
        results = {}
        player_points = {}
        while not self.gameOver:
            for currPlayer in self.playerQueue:
                numTurns += 1
                print("---------------------------------------------------------------------------")
                print("Current Player:", currPlayer.name)

                turnOver = False
                diceRolled = False

                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while not turnOver:
                    pygame.event.pump()
                    diceNum = self.rollDice()
                    print("Dice Rolled")
                    diceRolled = True
                    self.update_playerResources(diceNum, currPlayer)
                    self.diceStats[diceNum] += 1
                    self.diceStats_list.append(diceNum)

                    copyQueue = deque(copy.deepcopy(self.playerQueue))
                    copyQueue.rotate(-1 * (self.playerQueue.index(currPlayer, 0, self.numPlayers) + 1 % self.numPlayers))

                    if currPlayer.strategy == "mcts":
                        currPlayer.move(self.board, copyQueue)
                    elif currPlayer.strategy == "heuristic":
                        currPlayer.heuristic_move(self.board)

                    self.check_longest_road(currPlayer)
                    print(f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}")

                    self.boardView.displayGameScreen()
                    #pygame.time.delay(300) # no delay during experiments
                    turnOver = True

                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        turnOver = True
                        results = {
                            'winner': currPlayer.name,
                            'numTurns': int(numTurns / self.numPlayers),
                            'points': {player.name: player.victoryPoints for player in self.playerQueue},
                            'settlementsBuilt': {player.name: 5 - player.settlementsLeft for player in self.playerQueue},
                            'citiesBuilt': {player.name: 4 - player.settlementsLeft for player in self.playerQueue}
                        }
                        print("====================================================")
                        print(f"PLAYER {currPlayer.name} WINS IN {int(numTurns/self.numPlayers)} TURNS!")
                        print(self.diceStats)
                        break

                if self.gameOver:
                    startTime = pygame.time.get_ticks()
                    runTime = 0
                    while runTime < 5000:
                        runTime = pygame.time.get_ticks() - startTime

                    break
        return results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Settlers of Catan with AI Players")
    parser.add_argument('--numPlayers', type=int, default=2, help='Number of players (2 or 3)')
    parser.add_argument('--playerConfigs', type=str, default='[]', help='JSON string of player configurations')
    return parser.parse_args()

def main():
    args = parse_arguments()
    numPlayers = args.numPlayers
    playerConfigs = json.loads(args.playerConfigs)
    
    newGame_AI = catanAIGame(numPlayers=numPlayers, playerConfigs=playerConfigs)
    
    # Output results as JSON
    print(json.dumps(newGame_AI.results))

if __name__ == "__main__":
    main()

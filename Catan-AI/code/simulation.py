#Settlers of Catan
#Gameplay class with pygame with AI players

from board import *
from gameView import *
from player import *
from heuristicAIPlayer import *
import queue
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt

#Class to implement an only AI
class catanAIGame():
    #Create gameboard from current board in mcts.py
    def __init__(self, state):
        self.board = state["board"]

        #Game State variables
        self.gameOver = False
        self.maxPoints = 10
        self.numPlayers = 3
        self.player_name = state["current_player"].name
        self.result = -1
        #Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = state["queue"]

        return self.playCatan()
    

    #Function to initialize players + build initial settlements for players
    def build_initial_settlements(self):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            playerNameInput = input("Enter AI Player {} name: ".format(i+1))
            newPlayer = heuristicAIPlayer(playerNameInput, playerColors[i])
            newPlayer.updateAI()
            self.playerQueue.put(newPlayer)

        playerList = list(self.playerQueue.queue)

        #Build Settlements and roads of each player forwards
        for player_i in playerList: 
            player_i.initial_setup(self.board)
            pygame.event.pump()
            self.boardView.displayGameScreen()
            pygame.time.delay(1000)


        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList: 
            player_i.initial_setup(self.board)
            pygame.event.pump()
            self.boardView.displayGameScreen()
            pygame.time.delay(1000)
            
            print("Player {} starts with {} resources".format(player_i.name, len(player_i.setupResources)))

            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1
                    print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
        
        pygame.time.delay(5000)
        self.gameSetup = False


    #Function to roll dice 
    def rollDice(self):
        dice_1 = np.random.randint(1,7)
        dice_2 = np.random.randint(1,7)
        diceRoll = dice_1 + dice_2
        print("Dice Roll = ", diceRoll, "{", dice_1, dice_2, "}")

        return diceRoll

    #Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if(diceRoll != 7): #Collect resources if not a 7
            #First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            #print('Resources rolled this turn:', hexResourcesRolled)

            #Check for each player
            for player_i in list(self.playerQueue.queue):
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            print("{} collects 2 {} from City".format(player_i.name, resourceGenerated))

                print("Player:{}, Resources:{}, Points: {}".format(player_i.name, player_i.resources, player_i.victoryPoints))
                #print('Dev Cards:{}'.format(player_i.devCards))
                #print("RoadsLeft:{}, SettlementsLeft:{}, CitiesLeft:{}".format(player_i.roadsLeft, player_i.settlementsLeft, player_i.citiesLeft))
                print('MaxRoadLength:{}, Longest Road:{}\n'.format(player_i.maxRoadLength, player_i.longestRoadFlag))
        
        else:
            print("AI using heuristic robber...")
            currentPlayer.heuristic_move_robber(self.board)


    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in list(self.playerQueue.queue):
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

    #function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if(player_i.knightsPlayed >= 3): #Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue.queue):
                if(p.knightsPlayed >= player_i.knightsPlayed and p != player_i): #Check if any other players have more knights played
                    largestArmy = False
            
            if(largestArmy and player_i.largestArmyFlag == False): #if player_i takes largest army and didn't already have it
                #Set previous players flag to false and give player_i the largest points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.largestArmyFlag):
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))


    #Wrapper function to control all trading
    def trade(self, player_i):
        for r1, r1_amount in player_i.resources.items():
            if(r1_amount >= 6): #heuristic to trade if a player has more than 5 of a particular resource
                for r2, r2_amount in player_i.resources.items():
                    if(r2_amount < 1):
                        player_i.trade_with_bank(r1, r2)
                        break

    # function to simulate moves
    # TODO: implement with PPO
    def sim_move(self, board, player_i):
        self.trade(player_i)
        #Build a settlements, city and few roads
        possibleVertices = board.get_potential_settlements(player_i)
        if(possibleVertices != {} and (player_i.resources['BRICK'] > 0 and player_i.resources['WOOD'] > 0 and player_i.resources['SHEEP'] > 0 and player_i.resources['WHEAT'] > 0)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            player_i.build_settlement(list(possibleVertices.keys())[randomVertex], board)

        #Build a City
        possibleVertices = board.get_potential_cities(player_i)
        if(possibleVertices != {} and (player_i.resources['WHEAT'] >= 2 and player_i.resources['ORE'] >= 3)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            player_i.build_city(list(possibleVertices.keys())[randomVertex], board)

        #Build a couple roads
        for i in range(2):
            if(player_i.resources['BRICK'] > 0 and player_i.resources['WOOD'] > 0):
                possibleRoads = board.get_potential_roads(player_i)
                randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                player_i.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)

        #Draw a Dev Card with 1/3 probability
        devCardNum = np.random.randint(0, 3)
        if(devCardNum == 0):
            player_i.draw_devCard(board)

    #Function that runs the main game loop with all players and pieces
    def playCatan(self):
        #self.board.displayBoard() #Display updated board
        while (self.gameOver == False):
            #Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue.queue:
                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while(turnOver == False):
 
                    #Roll Dice and update player resources and dice stats
                    pygame.event.pump()
                    diceNum = self.rollDice()
                    diceRolled = True
                    self.update_playerResources(diceNum, currPlayer)

                    self.sim_move(self.board, currPlayer) #AI Player makes all its moves
                    #Check if AI player gets longest road and update Victory points
                    self.check_longest_road(currPlayer)
                    
                    self.boardView.displayGameScreen()#Update back to original gamescreen
                    pygame.time.delay(300)
                    turnOver = True
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        break

                if(self.gameOver):
                    if currPlayer.name == self.player_name:
                        self.result = 1
                    startTime = pygame.time.get_ticks()
                    runTime = 0
                    while(runTime < 5000): #5 second delay prior to quitting
                        runTime = pygame.time.get_ticks() - startTime

                    break
        return self.result
                                   
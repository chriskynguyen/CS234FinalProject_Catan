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
import copy

#Class to implement an only AI
class catanAIGame():
    #Create new gameboard
    def __init__(self):
        print("Initializing Settlers of Catan with only AI Players...")
        self.board = catanBoard(is_copy=False) # add copy functionality to board

        #Game State variables
        self.gameOver = False
        self.maxPoints = 10
        self.numPlayers = 0 

        #Dictionary to keep track of dice statistics
        self.diceStats = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        self.diceStats_list = []
        
        while(self.numPlayers not in [2,3]): # Only accept 2 and 3 player games, testing with 2
            try:
                self.numPlayers = int(input("Enter Number of Players (2 or 3):"))
            except:
                print("Please input a valid number")
        
        print("Initializing game with {} players...".format(self.numPlayers))
        print("Note that Player 1 goes first, Player 2 second and so forth.")
        
        #Initialize blank player queue and initial set up of roads + settlements
        #self.playerQueue = queue.Queue(self.numPlayers)
        self.playerQueue = deque()
        self.gameSetup = True #Boolean to take care of setup phase

        #Initialize boardview object
        self.boardView = catanGameView(self.board, self)

        #Function to go through initial set up
        self.build_initial_settlements()
        self.playCatan()

        #Plot diceStats histogram
        plt.hist(self.diceStats_list, bins = 11)
        plt.show()

        return None
    

    #Function to initialize players + build initial settlements for players
    def build_initial_settlements(self):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            playerNameInput = input("Enter AI Player {} name: ".format(i+1))
            usePPO = input("Use PPO (yes or no): ".format(i+1))
            exploration_param = input("Choose exploration parameter: ".format(i+1))
            newPlayer = heuristicAIPlayer(playerNameInput, usePPO, exploration_param, playerColors[i])
            newPlayer.updateAI()
            #self.playerQueue.put(newPlayer)
            self.playerQueue.append(newPlayer)

        playerList = list(self.playerQueue)

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
            for player_i in list(self.playerQueue):
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

                print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

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

                print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))



    #Function that runs the main game loop with all players and pieces
    def playCatan(self):
        #self.board.displayBoard() #Display updated board
        numTurns = 0
        while (self.gameOver == False):
            #Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue:
                numTurns += 1
                print("---------------------------------------------------------------------------")
                print("Current Player:", currPlayer.name)

                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while(turnOver == False):

                    #TO-DO: Add logic for AI Player to move
                    #TO-DO: Add option of AI Player playing a dev card prior to dice roll
                    
                    #Roll Dice and update player resources and dice stats
                    pygame.event.pump()
                    diceNum = self.rollDice()
                    print("Dice Rolled")
                    diceRolled = True
                    self.update_playerResources(diceNum, currPlayer)
                    self.diceStats[diceNum] += 1
                    self.diceStats_list.append(diceNum)

                    # copy queue for simulation
                    copyQueue = deque(copy.deepcopy(self.playerQueue)) 
                    copyQueue.rotate(-1 * (self.playerQueue.index(currPlayer, 0, self.numPlayers)+1 % self.numPlayers))
                    currPlayer.move(self.board, copyQueue) #AI Player makes all its moves
                    #Check if AI player gets longest road and update Victory points
                    self.check_longest_road(currPlayer)
                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))
                    
                    self.boardView.displayGameScreen()#Update back to original gamescreen
                    pygame.time.delay(300)
                    turnOver = True
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        print("====================================================")
                        print("PLAYER {} WINS IN {} TURNS!".format(currPlayer.name, int(numTurns/self.numPlayers)))
                        print(self.diceStats)
                        print("Exiting game in 10 seconds...")
                        pygame.time.delay(10000)
                        break

                if(self.gameOver):
                    startTime = pygame.time.get_ticks()
                    runTime = 0
                    while(runTime < 5000): #5 second delay prior to quitting
                        runTime = pygame.time.get_ticks() - startTime

                    break
                                   

#Initialize new game and run
newGame_AI = catanAIGame()
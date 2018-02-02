# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        if len(successorGameState.getFood().asList()) == 0: #Winstate
            return 9999999
        #if newPos == currentGameState.getPacmanPosition(): 
            #return -999999
        

        isFood = 0
        


        minGhostDist = 99999
        for ghost in newGhostStates: #Find distance to closest ghost
            if newPos == ghost.getPosition: #Penalize being eaten by ghosts
                return -999999
            dist = util.manhattanDistance(newPos, ghost.getPosition())
            if dist < minGhostDist:
                minGhostDist = dist
                closestGhostPos = ghost.getPosition()
        

        if newPos in currentGameState.getFood().asList(): #If pacman eats food
            return 1 + minGhostDist
            isFood = 1
        minFoodDist = 99999
        maxFoodDist = 0
        foodDistSum = 0
        for food in newFood.asList(): #Find min and max food dist
            dist = util.manhattanDistance(newPos, food)
            foodDistSum += dist
            if dist <= minFoodDist:
                minFoodDist = dist
        
        
        
        if minGhostDist > 3: #If ghost is more than 3 dist away, prioritize eating food
            return -minFoodDist + isFood
        else: #Otherwise prioritize avoiding ghosts
            return 2*minGhostDist -minFoodDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.val(gameState, 0, self.depth*gameState.getNumAgents())[1]
        util.raiseNotDefined()
    def val(self, gameState, agentIndex, depth):
        if agentIndex >= gameState.getNumAgents(): #Reset index to zero after all agents have gone
            agentIndex = 0
            depth = depth - 1
        if depth == 0: #Reached depth limit
            return [self.evaluationFunction(gameState)]
        actions = gameState.getLegalActions(agentIndex)
        states = [gameState.generateSuccessor(agentIndex, action) for action in actions]

        if len(states) == 0: #No legal actions
            return [self.evaluationFunction(gameState)]


        return self.miniMax(gameState, agentIndex, depth)




    def miniMax(self, gameState, agentIndex, depth):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0 or depth == 0: #No legal actions or depth limit reached
            return [self.evaluationFunction(gameState), Directions.STOP]
        maxVal = -999999
        minVal = 999999
        for action in actions: #For every possible action
            state = gameState.generateSuccessor(agentIndex, action)
            val = self.miniMax(state, agentIndex+1, depth-1)[0] 
            if agentIndex == 0: #Pacman, maximize score
                if val > maxVal:
                    res = [val, action]
                    maxVal = val
            else: #Ghost, minimize score
                if val < minVal:
                    res = [val, action]
                    minVal = val
        
        return res


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        return self.val(gameState, 0, self.depth*gameState.getNumAgents(), -999999, 999999)[1]
        util.raiseNotDefined()
    def val(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex >= gameState.getNumAgents(): #reset index to 0 when all agents have gone
            agentIndex = 0
            depth = depth - 1
        if depth == 0: #depth limit reached
            return [self.evaluationFunction(gameState)]
        actions = gameState.getLegalActions(agentIndex)
        states = [gameState.generateSuccessor(agentIndex, action) for action in actions]

        if len(states) == 0: #No legal actions
            return [self.evaluationFunction(gameState)]


        return self.alphaBeta(gameState, agentIndex, depth, alpha, beta)




    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex >= gameState.getNumAgents(): #Reset agent index to 0 when all agents have gone
            agentIndex = 0
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0 or depth == 0: #No legal actions or depth limit reached
            return [self.evaluationFunction(gameState), Directions.STOP]
        maxVal = -999999
        minVal = 999999
        for action in actions: #For all possible actions
            state = gameState.generateSuccessor(agentIndex, action)
            val = self.alphaBeta(state, agentIndex+1, depth-1, alpha, beta)[0] 
            if agentIndex == 0: #Pacman, maximize score
                if val > maxVal:
                    res = [val, action]
                    maxVal = val
                    if maxVal > beta: #AB Pruning
                        return res
                    alpha = max(maxVal, alpha) #Update alpha if new global max
            else: #Ghosts, minimize score
                if val < minVal:
                    res = [val, action]
                    minVal = val
                    if minVal < alpha: #AB Pruning
                        return res
                    beta = min(minVal, beta) #Update beta if new global min
        
        return res

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    



    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in actions]
        maxVal = -999999
        bestAction = Directions.STOP
        i = 0
        dep = (self.depth*gameState.getNumAgents())-1
        for state in states:
            #print "Checking action: ", actions[i]
            
            val = self.val(state, 1, dep)
            #print "Val is : ", val
            if val > maxVal:
                maxVal = val
                bestAction = actions[i]
            i = i + 1
        return bestAction

        util.raiseNotDefined()
    
    def val(self, gameState, agentIndex, depth):
        if agentIndex >= gameState.getNumAgents(): #reset index to 0 when last agent has gone
            agentIndex = 0
            #depth = depth - 1
        if depth == 0 or gameState.isWin() or gameState.isLose():
            #print "Final: ", self.evaluationFunction(gameState)
            return (self.evaluationFunction(gameState))
        if agentIndex == 0: #Pacman
            return self.maxValue(gameState, agentIndex, depth)
        else: #Ghosts
            return self.expectiValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0: #if no legal moves, return state val
            return self.evaluationFunction(gameState)
        states = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        maxVal = -999999
        i = 0
        for state in states:
            #print "Max action: ", actions[i]
            i = i + 1
            val = self.val(state, agentIndex+1, depth-1)
            if val > maxVal:
                maxVal = val
        return maxVal
    def expectiValue(self, gameState, agentIndex, depth):

        actions = gameState.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0: #if no legal moves, return state val
            return self.evaluationFunction(gameState)
        states = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        #print "testing"
        total = 0
        for state in states: #sum state vals, then return ave
            total = total + self.val(state, agentIndex+1, depth-1)
        
        #returns average successor state value
        return total/float(len(states))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
   
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    
    #If all food has been eaten
    if currentGameState.isWin():
        return 99999999

    dist = []
    ghostDist = []

    for food in foodList:
        dist.append(util.manhattanDistance(pos, food))

    for ghost in ghosts:
        ghostDist.append(util.manhattanDistance(pos, ghost.getPosition()))

    #Find min/max food/ghost distances
    minFoodDist = min(dist)
    maxFoodDist = max(dist)
    minGhostDist = min(ghostDist)
    maxGhostDist = max(ghostDist)

    #If Pacman is dead or ghost is very close
    if minGhostDist <= 1:
        return -999999
    
    #Weights
    foodWeight = -30
    minFoodDistWeight = -1
    minGhostDistWeight = -7

    #avoid ghost
    if minGhostDist < 4 and len(foodList) > 10: #Stay ~4 dist away from ghost until small amount of food left
        minGhostDistWeight = 15

    val = minFoodDistWeight*(minFoodDist) + minGhostDistWeight*minGhostDist + currentGameState.getScore() +foodWeight*len(foodList)
    return val
        

    


# Abbreviation
better = betterEvaluationFunction


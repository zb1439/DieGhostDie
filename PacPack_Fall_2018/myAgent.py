# myAgentP3.py
# ---------
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import Counter
from staffBot import SimpleStaffBot
import game
from util import nearestPoint


#########
# Agent #
#########
class MyAgent(CaptureAgent):
    """
    YOUR DESCRIPTION HERE
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        # Make sure you do not delete the following line.
        # If you would like to use Manhattan distances instead
        # of maze distances in order to save on initialization
        # time, please take a look at:
        # CaptureAgent.registerInitialState in captureAgents.py.
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.weight = Counter()
        self.weight['closestDot'] = 1
        self.weight['ghostPenalty'] = -0.5
        self.weight['friendPenalty'] = -1
        self.weight['numDots'] = 1


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        teammateActions = self.receivedBroadcast
        # Process your teammate's broadcast!
        # Use it to pick a better action for yourself

        currentAction = self.actionHelper(gameState)

        return currentAction

    def actionHelper(self, state):
        actions = state.getLegalActions(self.index)
        filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), state, self.index)

        val = float('-inf')
        best = None
        for action in filteredActions:
            new_state = state.generateSuccessor(self.index, action)
            new_state_val = self.evaluationFunction(new_state)

            if new_state_val > val:
                val = new_state_val
                best = action

        return best

    def getLimitedActions(self, state, index, remove_reverse=True):
        """
        Limit the actions, removing 'Stop', and the reverse action if possible.
        """
        actions = state.getLegalActions(index)
        actions.remove('Stop')

        if len(actions) > 1 and remove_reverse:
            rev = Directions.REVERSE[state.getAgentState(index).configuration.direction]
            if rev in actions:
                actions.remove(rev)

        return actions

    def evaluationFunction(self, state):
        foods = state.getFood().asList()
        ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
        friends = [state.getAgentPosition(pacman) for pacman in state.getPacmanTeamIndices() if pacman != self.index]

        pacman = state.getAgentPosition(self.index)

        closestFood = min(self.distancer.getDistance(pacman, food) for food in foods) + 2.0 \
            if len(foods) > 0 else 1.0
        closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) + 1.0 \
            if len(ghosts) > 0 else 1.0
        closestFriend = min(self.distancer.getDistance(pacman, friend) for friend in friends) + 1.0 \
            if len(friends) > 0 else 1.0

        closestFoodReward = 1.0 / closestFood
        closestGhostPenalty = 1.0 / (closestGhost ** 2) if closestGhost < 20 else 0
        if closestGhost < 3 : closestGhostPenalty += 1000
        closestFriendPenalty = 1.0 / (closestFriend ** 2) if closestFriend < 5 else 0

        numFood = len(foods)

        features = Counter()
        features['closestDot'] = closestFoodReward
        features['ghostPenalty'] = closestGhostPenalty
        features['friendPenalty'] = closestFriendPenalty
        features['numDots'] = numFood

        value = features * self.weight
        return value

class GameTreeAgent(MyAgent):

    """
    Currently HARDCODE to make it play with only one ghost
    for testing. The ghost and teammate may be multiple.
    @Comment: Zhibo
    """
    def chooseAction(self, gameState):
        self.treeLayer = [(self.index, None)]
        teammateActions = self.receivedBroadcast
        teamIndex = gameState.getPacmanTeamIndices()
        ghostIndex = gameState.getGhostTeamIndices()[0]
        for i in teamIndex:
            if i != self.index:
                teammateIndex = i

        #TODO: fix the teammateAction with the first legal action only and None iff nothing legal.
        if teammateActions and len(teammateActions) > 0:
            teammateAction = teammateActions.pop(0)
            if teammateAction in gameState.getLegalActions(teammateIndex):
                ghosts = [gameState.getAgentPosition(ghost) for ghost in gameState.getGhostTeamIndices()]
                pacman = gameState.getAgentPosition(teammateIndex)
                closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) \
                    if len(ghosts) > 0 else 1.0
                # If the ghost is nearby, re-plan based on game tree
                if closestGhost < 10:
                    self.treeLayer.append((teammateIndex, None))
                else:
                    self.treeLayer.append((teammateIndex, teammateActions))

        self.treeLayer.append((ghostIndex, None))
        self.terminal(gameState, 0, float('-inf'), float('inf'), saveMove=True)
        return self.decision


    def terminal(self, gameState, index, alpha, beta, saveMove=False):
        if index == len(self.treeLayer):
            return self.evaluationFunction(gameState)
        if self.treeLayer[index][0] in gameState.getPacmanTeamIndices():
            return self.maxValue(gameState, index, alpha, beta, saveMove)
        return self.minValue(gameState, index, alpha, beta, saveMove)

    def maxValue(self, gameState, index, alpha, beta, saveMove):
        v = float('-inf')
        agent, action = self.treeLayer[index]
        legalMoves = gameState.getLegalActions(agent)
        random.shuffle(legalMoves)
        decision = legalMoves[0]
        if action is None or len(action) == 0:
            allActions = gameState.getLegalActions(agent)
            allActions = actionsWithoutStop(allActions)
            allActions = actionsWithoutReverse(allActions, gameState, agent)
            for a in allActions:
                s = gameState.generateSuccessor(agent, a)
                newV = self.terminal(s, index + 1, alpha, beta)
                if v < newV:
                    v = newV
                    decision = a
                if v >= beta:
                    break
                alpha = max(v, alpha)
            self.decision = decision
            return v
        else:
            s = gameState.generateSuccessor(agent, action[0])
            v = self.terminal(gameState, index + 1, alpha, beta)
            self.decision = s
            return v

    def minValue(self, gameState, index, alpha, beta, saveMove):
        v = float('inf')
        agent, action = self.treeLayer[index]
        legalMoves = gameState.getLegalActions(agent)
        random.shuffle(legalMoves)
        decision = legalMoves[0]

        if action is None or len(action) == 0:
            allActions = gameState.getLegalActions(agent)
            allActions = actionsWithoutStop(allActions)
            allActions = actionsWithoutReverse(allActions, gameState, agent)
            for a in allActions:
                s = gameState.generateSuccessor(agent, a)
                newV = self.terminal(s, index + 1, alpha, beta)
                if v > newV:
                    v = newV
                    decision = s
                if v <= alpha:
                    break
                beta = min(v, beta)
            self.decision = decision
            return v
        else:
            s = gameState.generateSuccessor(agent, action[0])
            v = self.terminal(gameState, index + 1, alpha, beta)
            self.decision = s
            return v



def actionsWithoutStop(legalActions):
    """
    Filters actions by removing the STOP action
    """
    legalActions = list(legalActions)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    return legalActions


def actionsWithoutReverse(legalActions, gameState, agentIndex):
    """
    Filters actions by removing REVERSE, i.e. the opposite action to the previous one
    """
    legalActions = list(legalActions)
    reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
    if len(legalActions) > 1 and reverse in legalActions:
        legalActions.remove(reverse)
    return legalActions

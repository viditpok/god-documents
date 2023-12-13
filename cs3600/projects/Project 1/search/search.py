# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    stack = util.Stack()
    stack.push((problem.getStartState(), []))
    visited = set()

    while not stack.isEmpty():
        currentState, actions = stack.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited:
            visited.add(currentState)
            successors = problem.getSuccessors(currentState)
            for successor, action, _ in successors:
                if successor not in visited:
                    newActions = actions + [action]
                    stack.push((successor, newActions))

    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    queue = util.Queue()
    queue.push((problem.getStartState(), []))
    visited = set()

    while not queue.isEmpty():
        currentState, actions = queue.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited:
            visited.add(currentState)
            successors = problem.getSuccessors(currentState)
            for successor, action, _ in successors:
                if successor not in visited:
                    newActions = actions + [action]
                    queue.push((successor, newActions))

    return []



def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    priorityQueue.push((startState, []), 0)
    costSoFarar = {startState: 0}

    while not priorityQueue.isEmpty():
        currentState, actions = priorityQueue.pop()

        if problem.isGoalState(currentState):
            return actions

        successors = problem.getSuccessors(currentState)
        for successor, action, stepCost in successors:
            totalCost = costSoFarar[currentState] + stepCost

            if successor not in costSoFarar or totalCost < costSoFarar[successor]:
                costSoFarar[successor] = totalCost
                newActions = actions + [action]
                priorityQueue.push((successor, newActions), totalCost)

    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((problem.getStartState(), []), 0)
    visited = []
    
    while priorityQueue.isEmpty() == False:
        curr = priorityQueue.pop()
        if curr[0] not in visited:
            if problem.isGoalState(curr[0]):
                return curr[1]
            visited.append(curr[0])
            nodeSuccessors = problem.getSuccessors(curr[0])
            for x in nodeSuccessors:
                priorityQueue.push((x[0], curr[1] + [(x[1])]),
                          problem.getCostOfActions(curr[1] + [(x[1])]) +
                          heuristic(x[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

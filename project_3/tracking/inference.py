import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined
import util

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """


        if self.total() == 0.0:
            return

        tot = self.total()
        for key in self.keys():
            self[key] = float(self[key])/tot




    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """


        s_seq = []
        s_weights = []

        for item in self.items():
            s_seq.append(item[0])
            s_weights.append(float(item[1])/float(self.total()))

        x = random.random()

        for i, val in enumerate(s_seq):
            if x<=s_weights[i]:
                return val
            x-=s_weights[i]



class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference mod?ules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)


    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).

        You can use: busters.getObservationProbability(noisyDistance, trueDistance)
        to get the probability of seeing a particular sensor reading given some
        actual distance.

        Note 1: All distances in Pac-World are Manhattan!
        Note 2: The provided method does NOT handle the jail special cases.
                You must think those through and return appropriate values!
        """

        "*** YOUR CODE HERE ***"



        if ghostPosition == jailPosition and noisyDistance == None: 
            return 1
        elif ghostPosition != jailPosition and noisyDistance != None: 
            real_dist = manhattanDistance(pacmanPosition, ghostPosition) 
            observation_prob = busters.getObservationProbability(noisyDistance, real_dist)
            return observation_prob
        else: 
            return 0 



    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.  If a ghost is in jail, the sensor deterministically returns None.
        This is the only time the sensor will return None.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        
        
        curr_position = gameState.getPacmanPosition() 
        jail_position = self.getJailPosition()
        all_positions = self.allPositions
        
        for position in all_positions: 
            self.beliefs[position] = self.beliefs[position] * self.getObservationProb(observation, curr_position, position, jail_position)
        
        self.beliefs.normalize()


    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current state.
        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.

        Your agent has access to the action distribution for the ghost through
        self.getPositionDistribution.  In order to obtain the distribution over
        new positions for the ghost, given its previous position, use this line of code:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        Where oldPos refers to the previous ghost position.
        newPosDist is a DiscreteDistribution object, where for each position p
        in self.allPositions, newPosDist[p] is the probability that the ghost is
        at position p at time t + 1, given that the ghost is at position oldPos at time t

        """
        "*** YOUR CODE HERE ***"        
        
        
        all_positions = self.allPositions
        jail_position = self.getJailPosition()
        
        new_distribution = DiscreteDistribution()
        
        for old_position in all_positions: 
            if old_position not in jail_position: 
        
                newPosDist = self.getPositionDistribution(gameState, old_position)
                
                for new_position, prob in newPosDist.items(): 
                    new_distribution[new_position] += prob * self.beliefs[old_position]

        
        self.beliefs = new_distribution
        
        
        # util.raiseNotDefined()
        # python3 autograder.py -q q2 --no-graphics


    def getBeliefDistribution(self):
        return self.beliefs


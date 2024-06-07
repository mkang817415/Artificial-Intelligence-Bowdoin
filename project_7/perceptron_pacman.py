# perceptron_pacman.py
# --------------------
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.

        Return a list of predicted labels in the same order as the data.
        """
        guesses = []
        for datum, legalMoves in data:
            "*** YOUR CODE HERE ***"
            
            moves_score = util.Counter()
            
            for move in legalMoves: 
                moves_score[move] = self.weights * datum[move]
            
            movesPred = moves_score.argMax()
            guesses.append(movesPred)

        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                
                score_counter = util.Counter()
                
                label = trainingLabels[i]
                data = trainingData[i][0]
                legalMoves = trainingData[i][1]
                
                for move in legalMoves:
                    score_counter[move] = self.weights * data[move]
                
                movePred = score_counter.argMax()
                
                if movePred != label: 
                    self.weights -= data[movePred]
                    self.weights += data[label]


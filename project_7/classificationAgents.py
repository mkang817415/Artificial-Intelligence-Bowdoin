# classificationAgents.py
# -----------------------
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


# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent

import random
import game
import util

class DummyOptions:
    def __init__(self):
        self.data = "pacman"
        self.training = 25000
        self.test = 100
        self.odds = False
        self.weights = False


import perceptron_pacman
import neural_pacman

class ClassifierAgent(Agent):
    def __init__(self, trainingData=None, validationData=None, classifierType="perceptron", agentToClone=None, numTraining=3, early_stopping = None, learning_rate = None, hidden_neurons = None):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman, neuralFeatureExtractorPacman
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']
        self.featureFunction = enhancedFeatureExtractorPacman
        if(classifierType == "perceptron"):
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,numTraining)
        elif(classifierType == "neural"):
            classifier = neural_pacman.NeuralClassifierPacman(legalLabels,numTraining,
                               hidden_neurons = hidden_neurons,
                               early_stopping = early_stopping, learning_rate = learning_rate)
            self.featureFunction = neuralFeatureExtractorPacman
        elif(classifierType == "scikit-neural"):
            classifier = neural_pacman.ScikitNeuralClassifierPacman(legalLabels,numTraining)
            self.featureFunction = neuralFeatureExtractorPacman
        self.classifier = classifier
        args = {'featureFunction': self.featureFunction,
                'classifier':self.classifier,
                'printImage':None,
                'trainingData':trainingData,
                'validationData':validationData,
                'agentToClone': agentToClone,
        }
        options = DummyOptions()
        options.classifier = classifierType
        runClassifier(args, options)

    def getAction(self, state):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman, neuralFeatureExtractorPacman
        features = self.featureFunction(state)
        action =  self.classifier.classify([features])[0]
        return action


class NeuralClassifierAgent(ClassifierAgent):
    def __init__(self, trainingData=None, validationData=None, classifierType="neural", agentToClone=None, numTraining=1000):
        super().__init__(trainingData=trainingData, validationData=validationData, classifierType=classifierType, agentToClone=agentToClone, numTraining=numTraining)

class ScikitNeuralClassifierAgent(ClassifierAgent):
    def __init__(self, trainingData=None, validationData=None, classifierType="scikit-neural", agentToClone=None, numTraining=10000):
        super().__init__(trainingData=trainingData, validationData=validationData, classifierType=classifierType, agentToClone=agentToClone, numTraining=numTraining)


def scoreEvaluation(state):
    return state.getScore()

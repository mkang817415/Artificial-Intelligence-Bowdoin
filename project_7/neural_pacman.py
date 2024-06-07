# neural_pacman.py
# -------------
# This extension was written by David Byrd (d.byrd@bowdoin.edu).
# It adds feedforward neural network training to the Berkeley AI
# Pacman Perceptron project.
#
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

# For Q3 feature design only.  You may not use sklearn, or import any
# other library, to implement your Q4 network and backpropagation.
from sklearn.neural_network import MLPClassifier

# Neural implementation for apprenticeship learning
from sys import maxsize
from statistics import mean
import math
import random
import util
from pacman import GameState

PRINT = True
random.seed()

# Utility methods you might find useful.
def tanh(x):
    return math.tanh(x)

def tanh_grad(x):
    return 1 - math.tanh(x)**2


class NeuralClassifierPacman:
    """
    Feed-forward neural network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations, early_stopping = 10,
                        hidden_neurons = 5, learning_rate = 0.01):
        self.type = "neural"

        # Store init parameters.
        self.legalLabels = legalLabels
        self.num_labels = len(legalLabels)
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.num_hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate

        # Period for accuracy testing.
        self.period = 100

        # Early stopping condition.
        self.epsilon = 1e-5

        # Construct hidden and output neuron layers.
        # Output is a Counter of lists: label_name -> [ weights from hidden neurons ].
        # Output bias weights are a Counter: label_name -> bias weight at this neuron.
        # Hidden is a list of neurons, each a Counter: feature_name -> feature_weight.
        # There is a bias feature with value always -1, so we learn hidden bias "for free".
        
        self.output_neurons = util.Counter()
        self.output_bias_wt = util.Counter()
        self.hidden_neurons = [ util.Counter() for i in range(self.num_hidden_neurons) ]

        # Randomly initialize all output weights.
        for label in legalLabels:
            self.output_neurons[label] = [ random.random() - 0.5 for i in range(self.num_hidden_neurons) ]
            self.output_bias_wt[label] = random.random() - 0.5


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        Train the classifier on the provided data.  Validation data is not currently used.
        (But you certainly could!)

        trainingData is a list of (state features, legal actions) tuples.
        trainingLabels is a list of correct labels matching the trainingData in order.

        Nothing to return.  Just adjust the weights in the network.
        """

        # Break out the training data into two lists: the features and the legal actions
        # for each training data point.  These are two "parallel" lists.
        trainingData_state = [ x[0] for x in trainingData ]
        trainingData_legal = [ x[1] for x in trainingData ]

        # Extract the feature names from the first datum.
        # IMPORTANT: Your feature extractor must _always_ return all desired features.
        #            Your code will break if features are sometimes missing.
        self.features = trainingData_state[0].keys()

        # Now that we know the feature names, randomly initialize the hidden neurons.
        for h in self.hidden_neurons:
            for f in self.features:
                h[f] = random.random() - 0.5

        # These four variables are used to control early stopping.  We track the
        # list of recent SSE, the mean SSE of the most recent period, and the
        # number of consecutive periods without improvement greater than epsilon.
        period_sse = maxsize
        prev_sse = []
        no_improv = 0

        # Conduct the requested number of training iterations, unless we stop early.
        for iteration in range(self.max_iterations):

            # First you will need to run the network forward on all of the training
            # data to get hidden and output activation levels, and also turn the
            # output activations into y predictions.
            h_acts, o_acts = self.forward(trainingData_state)
            guesses = self.classify(trainingData, o_acts)

            # Track the SSE for this entire iteration.
            sse = 0

            # For each training datum, iterate in parallel through...
            # x_features: the feature Counter for this state,
            # guess: the predicted output label,
            # h_a, o_a: the hidden and output activations for this state,
            # answer: the correct label for this state.

            for x_features, guess, h_a, o_a, answer in zip(trainingData_state,
                                       guesses, h_acts, o_acts, trainingLabels):

                # Backpropagation!  My working implementation is 13 lines of Python.
                # I used no libraries, no additional imports, just our Counter class
                # and purely iterative code, handling one weight at a time.

                "*** YOUR CODE HERE ***"

                util.raiseNotDefined()


            if (iteration > 0) and (iteration % self.period == 0):
                # Keep last period sse only.
                prev_sse = prev_sse[-self.period:]
                prev_period_sse = period_sse
                period_sse = mean(prev_sse)

                # Print stats.  How is it going?
                self.check_accuracy(iteration, period_sse, trainingData, trainingLabels)

                # Check for early stopping.
                if period_sse < prev_period_sse - self.epsilon: no_improv = 0
                else: no_improv += 1

                # Stop if enough periods in a row don't show sufficient improvement.
                if no_improv >= self.early_stopping: break

            # Record the SSE from this iteration.
            prev_sse.append(sse)


    def forward(self, data):
        """
        Data is a list of each datum to run forward through the network.
        Datum is a Counter representing the features of a GameState.
        """

        # Must return two lists.  Each top-level list entry is the
        # activations for one datum.

        # For hidden, each entry is a list of length num_hidden_neurons,
        #             each neuron having one real-valued activation.
        # For output, each entry is a Counter with num_output_neurons entries,
        #             each one: output_label_name -> real-valued activation.
        data_hidden_activations = []
        data_output_activations = []

        # Run one input datum forward through the network and record the results.
        for datum in data:

          # Accumulate hidden activations for this datum.
          hidden_activations = []

          # Accumulate output activations for this datum.
          output_activations = util.Counter()

          # Do the forward network calculation.  My implementation had 11 lines.

          " *** YOUR CODE HERE *** "

          util.raiseNotDefined()

        return data_hidden_activations, data_output_activations


    def classify(self, data, activations = None ):
        """
        Data contains a list of (input datum, legal moves).

        If forward has already been run, activations can be passed
        the resulting output activations in the same order as "data".
        This will avoid redundant calculation.  If activations is
        None, a forward pass must be run inside classify to get the
        output activations.
        
        Datum is a Counter representing the features of each GameState.
        Legal moves is a list of legal moves for that GameState.
        """
        data_state = [ x[0] for x in data ]
        data_legal = [ x[1] for x in data ]

        # We must return a list of the output labels predicted for
        # each datum.
        output_class_predictions = []

        # NOTE: The Pacman data has some capitalization issues for labels
        #       (actions).  Your output neurons are probably, e.g. "WEST",
        #       but the legal moves are, e.g. "West".  Using Python's
        #       string.capitalize() will help!

        # HINT: Your network may give the highest output activation to
        #       an illegal move!  Be sure to select the LEGAL move that
        #       has the highest output activation strength.

        # My classify implementation was 7 lines.

        " *** YOUR CODE HERE *** "

        util.raiseNotDefined()

        return output_class_predictions


    def check_accuracy(self, iter, period_sse, trainingData, trainingLabels):
        """
        Classify the training data and print stats to show how things are going.
        """

        guesses = self.classify(trainingData)
        correct = [guesses[i] == trainingLabels[i] for i in range(len(trainingLabels))].count(True)
        acc_str = f"IS Acc: {correct} out of {len(trainingLabels)} " + \
                  f"({100.0 * correct / len(trainingLabels):.1f}%)"

        print (f"Iteration {iter}... SMA-{self.period} SSE: {period_sse:.4f}.  {acc_str}.")


class ScikitNeuralClassifierPacman:
    """
    Pre-built neural classifier for feature design.
    """
    def __init__( self, legalLabels, max_iterations, early_stopping = 10,
                        hidden_neurons = 40, learning_rate = 0.01):
        self.legalLabels = legalLabels
        self.type = "neural"
        self.max_iterations = max_iterations
        self.num_hidden_neurons = hidden_neurons
        self.num_labels = len(legalLabels)

        self.clf = MLPClassifier(max_iter=200, hidden_layer_sizes=(self.num_hidden_neurons),
                                 tol=1e-3, activation='tanh')

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        Magic training with scikit-learn (sklearn).
        """

        # Get the feature names.
        trainingData = [ x[0] for x in trainingData ]
        self.features = trainingData[0].keys()

        # Map actions to neuron numbers.
        label_map = {'North':0, 'South':1, 'East':2, 'West':3, 'Stop':4}

        # Shape the training data for scikit.
        X_train = [ [ d[x] for x in d ] for d in trainingData ]
        y_train = [ label_map[y] for y in trainingLabels ]

        # Train using scikit (ignore convergence warnings).
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.clf.fit(X_train, y_train)

    def forward(self, data ):
        """
        Computes activation strength for each output neuron for each datum.
        Not needed for scikit approach.  Equivalent to self.clf.predict_proba(X_test).
        """
        util.raiseNotDefined()
        return


    def classify(self, data ):
        """
        Computes output class prediction for each datum using scikit-learn.
        """
        data_legal = [ x[1] for x in data ]
        data_state = [ x[0] for x in data ]

        # Map output neuron numbers back to action labels.
        label_map = {0:'North', 1:'South', 2:'East', 3:'West', 4:'Stop'}

        # Shape the input data states for sklearn.
        X_test = [ [ d[x] for x in d ] for d in data_state ]

        # Cannot use self.clf.predict, because it does not know that the
        # allowable actions differ for every datum.  Get raw output
        # values per action instead.
        y_test_raw = self.clf.predict_proba(X_test)

        # Manually determine the best legal action for each datum.
        y_test = []
        for o_acts, o_legal in zip(y_test_raw, data_legal):
            best, arg_best = -maxsize, None

            for i, act in enumerate(o_acts):
                if act > best and label_map[i] in o_legal:
                    best = act
                    arg_best = i

            y_test.append(label_map[arg_best])

        return y_test


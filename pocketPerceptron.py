import numpy as np
import pandas as pd
import collections, random, copy


class Perceptron():
    def __init__(self, numInputs, epochs=100, learningRate=0.01, rand=False, threshold=0.5):
        # initial weights can be a random vector or a zero vector
        if rand:
            self.weights = np.random.rand(numInputs + 1)
        else:
            self.weights = np.zeros(numInputs + 1)
        
        self.epochs = epochs
        self.learningRate = learningRate
        self.threshold = threshold

        # saves a copy of the initial weights
        self.initialWeights = copy.deepcopy(self.weights)

    def predict(self, inputs, weights=""):
        # predicts yes or no based on given inputs and weights

        # if no weights are given, default is self.weights
        if type(weights) == type(str()):
            weights = self.weights

        # sum of inputs and weights plus bias, which is represented by weights[0]
        self.sum = np.dot(inputs, weights[1:]) + weights[0]

        # if sum > threshold, neuron activates
        if self.sum > self.threshold:
            activation = 1
        else:
            activation = 0

        return np.float64(activation)

    def correctClassifications(self, weights):
        # returns the number of correct classifications over all training inputs for a given weight vector

        classifications = 0

        for input, label in zip(self.trainingInputs, self.labels):
            if self.predict(input, weights) == label:
                classifications += 1

        return classifications

    def train(self, trainingInputs, labels, verbose=False):
        self.trainingInputs = trainingInputs
        self.labels = labels
        self.verbose = verbose

        # initialize the pocket, and add in initial weights and correctClassifications of initial weights
        pocket = []
        pocket.append(self.initialWeights)
        pocket.append(self.correctClassifications(self.initialWeights))

        if self.verbose:
            print("initial weights:", self.initialWeights)
            print("pocket weights:", pocket[0])
            print("pocket classifications:", pocket[1])

        epoch = 0
        # correct classifications made by weight vector in pocket
        bestRunLength = 0
        currentRunLength = 0

        while (epoch < self.epochs):
            epoch += 1
            if self.verbose:
                print("\ncurrent epoch:", epoch)
                print("pocket weights:", pocket[0])
                print("pocket classifications:", pocket[1])
                print("current weights:", self.weights)

            # correct classifications made by weight vector in current iteration
            randInputVector = random.sample(
                list(zip(self.trainingInputs, self.labels)), 1)
            input = randInputVector[0][0]
            label = randInputVector[0][1]
            prediction = self.predict(input, self.weights)
            if self.verbose:
                print("random input vector:", input)
                print("label:", label)
                print("prediction:", prediction)
                print("label == prediction:", label == prediction)

            if label != prediction:
                # prediction was incorrect: compare pocket weight vector against current weight 
                # vector (with possible replacement), reset current run and adjust weights

                if self.verbose:
                    print(" WRONG PREDICTION "*4)
                    print("best run length: %f" % (bestRunLength))
                    print("current run length: %f" % (currentRunLength))
                    print("current correct classifications: %f" %(self.correctClassifications(self.weights)))
                    print("pocket correct classifications: %f" % (pocket[1]))

                # if the current run is longer then the best run AND the current weights misclassify 
                # less points then the pocket weights, put the current weights in the pocket and 
                # update the best run with the current run

                if (currentRunLength > bestRunLength):
                    if (self.correctClassifications(self.weights) > pocket[1]):
                        if self.verbose:print(" POCKET CHANGED "*4)
                        pocket[0] = self.weights
                        pocket[1] = self.correctClassifications(pocket[0])
                        bestRunLength = currentRunLength

                if self.verbose:
                    print("adjusting weights...")
                
                # adjust weights
                self.weights[1:] += self.learningRate * (label - prediction) * input
                self.weights[0] += self.learningRate * (label - prediction)

                currentRunLength = 0
            else:
                # the prediction was correct: increment run length
                currentRunLength += 1

                if self.verbose:
                    print("best run length: %f" % (bestRunLength))
                    print("current run length: %f" % (currentRunLength))
                    print("current correct classifications: %f" %
                          (self.correctClassifications(self.weights)))
                    print("pocket correct classifications: %f" % (pocket[1]))

            if self.verbose:
                print("current weights:", self.weights)
                print("pocket weights:", pocket[0])
                print("pocket classifications:", pocket[1])

        if self.verbose:
            print("\ninitial weights:", self.initialWeights)
            print("initial weight classifications:", self.correctClassifications(self.initialWeights))
            print("final pocket weights:", pocket[0])
            print("final pocket classifications:", pocket[1])


def cleanData(pathToData, dataType, flower = None):
    # Cleans up training data. Ensures equal number of points for each label and replaces label name with a vector

    # import data as pandas DataFrame
    with open(pathToData, "r") as file:
        df = pd.read_csv(file, names=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

    # ensure we have same number of data points for each label
    numLabels = collections.Counter(list(df.species))
    if len(set(numLabels.values())) != 1:
        raise ValueError("Disproportionate number of data points for each class")


    # data is returned differently based on requirement 
    if dataType == "training":
        if flower == None:
            raise ValueError("No flower specified")

        # replace label names with label vectors
        a, b, c = 0, 0, 0
        labels = {"Iris-setosa": a, "Iris-versicolor": b, "Iris-virginica": c}
        labels[flower] = 1
        df.species = df.species.map(labels)
    elif dataType == "test":
        labels = {"Iris-setosa": (1, 0, 0), "Iris-versicolor": (0,1, 0), "Iris-virginica": (0, 0, 1)}
        df.species = df.species.map(labels)
    else:
        raise ValueError("Wrong dataType specified")

    # shuffle data so its not ordered
    df = df.sample(frac=1).reset_index(drop=True)

    # returns a tuple of the form (input vector, associated label)
    return df.iloc[:, 0:-1].values, df.iloc[:, -1].values


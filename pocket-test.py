from pocketPerceptron import Perceptron, cleanData
import numpy as np
import pandas as pd

import sys

# set parameters
epochs = 500
learningRate = 0.01
randomWeights = True
verbose = False

groups = {k: None for k in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]}
for group in groups:
    # import training data
    train_data, train_labels = cleanData("iris_train.txt", "training", group)

    # initialize perceptrons
    perceptron = Perceptron(4, epochs, learningRate, randomWeights)

    print(f"Training {group} perceptron...")

    # train perceptrons
    perceptron.train(train_data, train_labels, verbose)
    print(f"initial weights: {perceptron.initialWeights}")
    print(f"final weights: {perceptron.weights}")

    groups[group] = perceptron

# import test data
irisTestData, irisTestLabels = cleanData("iris_test.txt", "test")

outputs = []
success = []
wrong = 0
possibleLabels = [(1, 0, 0), (0, 1, 0), (1, 0, 0)]
labelsToVector = {
    "Iris-setosa": (1, 0, 0),
    "Iris-versicolor": (0, 1, 0),
    "Iris-virginica": (0, 0, 1),
}
vectorToLabels = {
    (1, 0, 0): "Iris-setosa",
    (0, 1, 0): "Iris-versicolor",
    (0, 0, 1): "Iris-virginica",
}

for flower, label in zip(irisTestData, irisTestLabels):
    outputVector = [0, 0, 0]

    outputVector = [v.predict(flower) for v in groups.values()]
    listOfSums = [v.sum for v in groups.values()]

    maxIndex = np.argmax(listOfSums)

    if outputVector == [0, 0, 0]:
        # edge case where only one perceptron fires
        # chooses the perceptron with highest sum value, i.e. higher confidence
        outputVector[maxIndex] = 1
    elif outputVector.count(1) > 1:
        # edge case where more than one perceptron fires
        # chooses the perceptron with highest sum value, i.e. higher confidence
        a = [0, 0, 0]
        for i in range(len(outputVector)):
            if outputVector[i] == 1:
                a[i] = listOfSums[i]

        outputVector = [0, 0, 0]
        outputVector[np.argmax(a)] = 1

    outputVector = tuple(np.float64(outputVector))

    predictionCorrect = outputVector == label
    success.append(predictionCorrect)
    outputs.append(outputVector)

    # results.append([label, outputVector, success])
    if not predictionCorrect:
        wrong += 1

## converts results to dataFrame
results = pd.DataFrame(
    {"labels": irisTestLabels, "prediction": outputs, "correctness": success}
)

## confusion matrix

# initialize confusion matrix
confusionMatrix = np.zeros((3, 3))

# assign values to confusion matrix with predicted vals on x axis and true vals on y axis
for i, j in zip(irisTestLabels, outputs):
    iter = [0, 1, 2]
    for a, b in zip(iter, iter):
        if i == possibleLabels[a] and j == possibleLabels[b]:
            confusionMatrix[a, b] += 1


print("confusion matrix ( predicted vals on x axis and true vals on y axis):")
print(confusionMatrix)

# calculates accuracy
correctPercentage = 100 * (30 - wrong) / 30
print("Correctly classified %.0f%%" % (correctPercentage))

# convert predicted vectors to labels
for i in range(len(outputs)):
    outputs[i] = vectorToLabels[outputs[i]]

# combine labels and data points, then output to txt file
pd.concat([pd.DataFrame(irisTestData), pd.DataFrame(outputs)], axis=1).to_csv(
    "iris_output.txt", index=False
)

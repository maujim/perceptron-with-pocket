from pocketPerceptron import Perceptron, cleanData
import numpy as np
import pandas as pd

import sys

# import training data
irisSetosaTrainData, irisSetosaTrainLabels = cleanData("iris_train.txt", "training", "Iris-setosa")
irisVersicolorTrainData, irisVersicolorTrainLabels = cleanData("iris_train.txt", "training", "Iris-versicolor")
irisVirginicaTrainData, irisVirginicaTrainLabels = cleanData("iris_train.txt", "training", "Iris-virginica")

# import test data
irisTestData, irisTestLabels = cleanData("iris_test.txt", "test")

# set parameters
epochs = 250
learningRate = 0.01
randomWeights = True
verbose = False

# initialize perceptrons
perceptronSetosa = Perceptron(4, epochs, learningRate, randomWeights)
perceptronVersicolor = Perceptron(4, epochs, learningRate, randomWeights)
perceptronVirginica = Perceptron(4, epochs, learningRate, randomWeights)

# train perceptrons
if verbose: print("Training setosa perceptron...")
perceptronSetosa.train(irisSetosaTrainData, irisSetosaTrainLabels, verbose)
print("initial weights:")
print(perceptronSetosa.initialWeights)
print("final weights:")
print(perceptronSetosa.weights)

if verbose: print("Training veriscolor perceptron...")
perceptronVersicolor.train(irisVersicolorTrainData, irisVersicolorTrainLabels, verbose)
print("initial weights:")
print(perceptronVersicolor.initialWeights)
print("final weights:")
print(perceptronVersicolor.weights)

if verbose: print("Training virginica perceptron...")
perceptronVirginica.train(irisVirginicaTrainData, irisVirginicaTrainLabels, verbose)
print("initial weights:")
print(perceptronVirginica.initialWeights)
print("final weights:")
print(perceptronVirginica.weights)

outputs = []
success = []
wrong = 0
possibleLabels = [(1,0,0), (0,1,0), (1,0,0)]
labelsToVector = {"Iris-setosa": (1, 0, 0), "Iris-versicolor": (0,1, 0), "Iris-virginica": (0, 0, 1)}
vectorToLabels = {(1, 0, 0): 'Iris-setosa', (0, 1, 0): 'Iris-versicolor', (0, 0, 1): 'Iris-virginica'}

for flower, label in zip(irisTestData, irisTestLabels):
    outputVector = [0,0,0]
    
    outputVector = [
        perceptronSetosa.predict(flower),
        perceptronVersicolor.predict(flower),
        perceptronVirginica.predict(flower)
    ]

    listOfSums = [perceptronSetosa.sum, perceptronVersicolor.sum, perceptronVirginica.sum]

    maxIndex = np.argmax(listOfSums)

    if outputVector == [0,0,0]:
    # edge case where only one perceptron fires
    # chooses the perceptron with highest sum value, i.e. higher confidence
        outputVector[maxIndex] = 1
    elif outputVector.count(1) > 1:
    # edge case where more than one perceptron fires
    # chooses the perceptron with highest sum value, i.e. higher confidence
        a = [0,0,0]
        for i in range(len(outputVector)):
            if outputVector[i] == 1:
                a[i] = listOfSums[i]

        outputVector = [0,0,0]
        outputVector[np.argmax(a)] = 1

    outputVector = tuple(np.float64(outputVector))

    predictionCorrect = (outputVector == label)
    success.append(predictionCorrect)
    outputs.append(outputVector)

    # results.append([label, outputVector, success])
    if not predictionCorrect:
        wrong += 1

## converts results to dataFrame
results = pd.DataFrame(
    {
        "labels": irisTestLabels,
        "prediction": outputs,
        "correctness": success
    }
)

## confusion matrix

# initialize confusion matrix
confusionMatrix = np.zeros((3,3))

# assign values to confusion matrix with predicted vals on x axis and true vals on y axis
for i,j in zip(irisTestLabels, outputs):
    if i == possibleLabels[0]:
        if j == possibleLabels[0]:
            confusionMatrix[0,0] += 1
        if j == possibleLabels[1]:
            confusionMatrix[0,1] += 1
        if j == possibleLabels[2]:
            confusionMatrix[0,2] += 1
    if i == possibleLabels[1]:
        if j == possibleLabels[0]:
            confusionMatrix[1,0] += 1
        if j == possibleLabels[1]:
            confusionMatrix[1,1] += 1
        if j == possibleLabels[2]:
            confusionMatrix[1,2] += 1
    if i == possibleLabels[2]:
        if j == possibleLabels[0]:
            confusionMatrix[2,0] += 1
        if j == possibleLabels[1]:
            confusionMatrix[2,1] += 1
        if j == possibleLabels[2]:
            confusionMatrix[2,2] += 1

print("confusion matrix ( predicted vals on x axis and true vals on y axis):")
print(confusionMatrix)

# calculates accuracy
correctPercentage = 100*(30-wrong)/30
print("Correctly classified %.0f%%" %(correctPercentage))

# convert predicted vectors to labels
for i in range(len(outputs)):
    outputs[i] = vectorToLabels[outputs[i]]

# combine labels and data points, then output to txt file
pd.concat([pd.DataFrame(irisTestData), pd.DataFrame(outputs)], axis=1).to_csv("iris_output.txt",index=False)

print(initi)
# perceptron-with-pocket

assignment 1 for cmpe 452, october 2019

## Instructions to run code

Run `pip install -r requirements.txt`. Ensure both the training and test datasets are in the same directory and then run `python pocket-test.py`.

## Structure

This is a two-layer ANN with one output and one input layer. The former has 3 nodes and the latter has 5 nodes, 4 of which are inputs and the fifth being a “dummy” node to accommodate for the bias being represented as a weight.

The confusion matrix shall be printed in the code itself and the output file required shall be printed to “iris_output.txt”. Initial and final weight vectors are also printed in the code itself. The code runs for 500 iterations. 

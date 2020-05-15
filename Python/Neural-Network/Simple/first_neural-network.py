# -*- coding: utf-8 -*-
"""
Created on Wed 17 Oct

@author: MKY01

Reference: https://www.youtube.com/watch?v=gwitf7ABtK8
"""

import numpy as np
import os

#Neural Network
"""This Algorithm is the implementation of a simple artificial neural network,
which consists of: an input layer, output layer, a set of Weights and Biases.
"""
def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

#Data set
"""This is the set of fictional training data used (in an array)"""

data = [[2, 2.5, 1], [3, 1.5, 0], [5, 2, 1], [4, 2, 1], [3.5, 2, 0], [2.5, 1.5, 0], [6, 1.5, 1], [3, 2, 0]]


#Bot
"""This is a basic implementation of pre-defined phrases (strings) in conjunction with the Algorithm above,
to try to make a prediction"""

phrases = ['I think', 'I guess', 'I predict', 'Looks like', 'Maybe it is', 'seems like its']

rand_data = data[np.random.randint(len(data))]

m1 = rand_data[0]
m2 = rand_data[1]

prediction = NN(m1, m2, w1, w2, b)
prediction_text = ["White", "Black"][int(np.round(prediction))]
phrase = np.random.choice(phrases) + " " + prediction_text
print(phrase)

o = os.system("say " + phrase)

print("It's actually " + ["White", "Black"][rand_data[2]])

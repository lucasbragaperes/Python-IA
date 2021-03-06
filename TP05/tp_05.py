# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:19:36 2021

@author: FATEC_Lucas
"""

from linear_algebra import dot
import math
import random

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    return 1/ (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []
    
    for layer in neural_network:
        
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        
        input_vector = output
        
    return outputs

def backpropagate(network, input_vector, target):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    output_deltas = [output * (1-output)* (output - target[i])
                     for i, output in enumerate(outputs)]
    
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
            
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input
            
if __name__ == "__main__":
    
    raw_digits = [
        """11111
           1...1
           11111
           1...1
           1...1""",
           
        """11111
           1....
           11111
           1....
           11111""",
           
        """..1..
           ..1..
           ..1..
           ..1..
           ..1..""",
           
        """11111
           1...1
           1...1
           1...1
           11111""",
           
        """1...1
           1...1
           1...1
           1...1
           11111""",]
           
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]
    
    inputs = list(map(make_digit, raw_digits))
    
    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]
    
    random.seed(0)
    input_size = 25
    num_hidden = 5
    output_size = 5
    
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]
    
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]
    
    network = [hidden_layer, output_layer]
    
    # treinamento
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)
            
    def predict(input):
        return feed_forward(network, input)[-1]
    
    for i, input in enumerate(inputs):
        outputs = predict(input)
        
        def switch(i):
            switcher = {
                0: 'A',
                1: 'E',
                2: 'I',
                3: 'O',
                4: 'U'}
            
            return switcher.get(i)
        
        print(switch(i), [round(p,2) for p in outputs])
        
# teste de varia????es n??o treinadas
    
    print("""@@@@@
@...@
@@@@@
@...@
@...@""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,     # .111.
                     1,0,0,0,1,     # 1...1
                     1,1,1,1,1,     # 11111
                     1,0,0,0,1,     # 1...1
                     1,0,0,0,1])])  # 1...1
    
    print("""@@@@@
@....
@@@@@
@....
@@@@@""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,     # .111.
                     1,0,0,0,0,     # 1....
                     1,1,1,1,1,     # 11111
                     1,0,0,0,0,     # 1....
                     0,1,1,1,0])])  # .111.
    print("""..@..
..@..
..@..
..@..
..@..""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,     # .111.
                     0,0,1,0,0,     # ..1..
                     0,0,1,0,0,     # ..1..
                     0,0,1,0,0,     # ..1..
                     0,1,1,1,0])])  # .111.
    print("""@@@@@
@...@
@...@
@...@
@@@@@""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,     # .111.
                     1,0,0,0,1,     # 1...1
                     1,0,0,0,1,     # 1...1
                     1,0,0,0,1,     # 1...1
                     0,1,1,1,0])])  # .111.
    print("""@...@
@...@
@...@
@...@
@@@@@""")
    print([round(x, 2) for x in
           predict( [1,0,0,0,1,     # 1...1
                     1,0,0,0,1,     # 1...1
                     1,0,0,0,1,     # 1...1
                     1,0,0,0,1,     # 1...1
                     0,1,1,1,0])])  # .111.

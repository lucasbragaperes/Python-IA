# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:12:40 2021

@author: FATEC_Lucas
"""

from __future__ import division 
from collections import Counter 
from functools import partial 
from linear_algebra import dot 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math #pg 56 - Summerfield 
import math, random 

def saida_adaline(pesos, entradas): 
    y = dot(pesos, entradas)
    return y 

def sigmoid(t): 
    return ((2 / (1 + math.exp(-t)))-1)

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []
    
    for layer in neural_network: 
        
        input_with_bias = input_vector + [1]                  # adiciona bias à entrada
        output = [neuron_output (neuron, input_with_bias)     # calcula a saída do neurônio
            for neuron in layer]                              # para cada camada
        #print (output)                                 
        outputs.append(output)                                #acrescenta valor à saída memorizando


        # a saída de uma camada de neurônio é a entrada da próxima camada
        input_vector = output 
        
    return outputs

alpha = 0.1
def backpropagate(network, input_vector, target):
    # feed_forward calcula a saida dos neuronios usando sigmóide
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # 0.5 * alpha* (1 + output) * (1 - output) calculo da derivada da sigmoide
    output_deltas = [0.5 * (1 + output) * (1 - output) * (output - target[i]) * alpha
            for i, output in enumerate (outputs)]

    #ajuste dos pesos sinápticos para camada de saída (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    #0.5 * alpha* (1 + output) * (1 - output) calculo da derivada da sigmoide
    #retro propagacao do erro para camadas intermediarias
    hidden_deltas = [ 0.5 * alpha * (1 + hidden_output) * (1 - hidden_output) * dot (output_deltas, [n[i] for n in network [-1]])
        for i, hidden_output in enumerate(hidden_outputs)]
    
    #ajuste dos pesos sinápticos para camadas intermediarias (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def seno(x): #funcao a ser aproximada pela rede neural
    seno = [0.8+(math.sin(math.pi/180*x)*math.sin(2*math.pi/180*x))*0.5]
    return [seno]

def predict(inputs):
    return feed_forward(network, inputs)[-1]

inputs = []
targets = []    
for x in range(360):
    seno_a = seno(x)

random.seed(0)
input_size = 1
num_hidden = 8
output_size = 1

hidden_layer = [[random.random() for __ in range(input_size + 1)]
        for __ in range(num_hidden)]

output_layer = [[random.random() for __ in range(num_hidden + 1)]
        for __ in range(output_size)]

network = [hidden_layer, output_layer]

for __ in range(500):
    for x in range(360):
        inputs = seno(x)
        targets = seno(x)
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

# formatação do gráfico
fig,ax = plt.subplots()
ax.set(xlabel='ângulo (°)', ylabel='função sen(x)*sen(2x)',
title='Aproximaçao Funcional')
ax.grid()
t=np.arange(0,360,1)
# teste de rede através de predict()
saida=[]
for x in range(360):
    inputs = seno(x)
    targets = seno(x)
    for input_vector, target_vector in zip(inputs, targets):
        sinal_saida = predict(input_vector)
    saida.extend(sinal_saida)

entrada=[]
for x in range (360):
    entrada += seno(x) #criando o arranjo da função de entrada para o gráfico
ax.plot(t, entrada)
ax.plot(t, saida)
plt.show()

print ("camada entrada", hidden_layer)
print ("camada saída", output_layer)
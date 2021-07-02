# -*- coding: utf-8 -*-
"""
FATEC - Análise e Desenvolvimento de Sistemas
TP01 - Inteligência Artificial

Lucas Braga Peres
Steffanie Graner

"""
from typing import List

Vector = List[float]

def dot(v: Vector, w: Vector) -> float:
    "Computes v_1 * w_1 + ... + v_n * w_n"
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
"Função Dot"

def step_function(x):
    return 1 if x == 0 else 0 
"Degrau da função"

def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    "Somatório"
    return step_function(calculation)

x0 = [0,0]
x1 = [0,1]
x2 = [1,0]
x3 = [1,1]
"Entradas"

weights = [1,1]
"Pesos"
bias = 0 

saida0 = perceptron_output(weights, bias, x0)
saida1 = perceptron_output(weights, bias, x1)
saida2 = perceptron_output(weights, bias, x2)
saida3 = perceptron_output(weights, bias, x3)

print("PERCEPTRON IMPLEMENTANDO FUNÇÃO BOOLEANA NOU (NOR)")
print("0 AND 0 -", saida0)
print("0 AND 1 -", saida1)
print("1 AND 0 -", saida2)
print("1 AND 1 -", saida3)

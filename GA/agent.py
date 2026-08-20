import numpy as np
import torch

from Teste_IA.SnakeGameAI.Geral.snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet  


# Paramentros geneticos
MUTATION_RATE = 0.15 # % dos pesos que sofrem mutação em cada filho
MUTATION_STRENGTH = 0.30 # Desvio padrão do ruído gaussiano aplicado
MAX_STEPS_WITHOUT_FOOD = 100

class Agent:
    def __init__(self, model: Linear_QNet = None):
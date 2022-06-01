import linecache
import os
import logging
from app.modules.puissance4.Games import Connect4
from app.modules.puissance4.MCTS import MCTS

from app.modules.puissance4.Models import NeuralNetwork

SPECIFICATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
GPU_MEMORY_LIMIT = int(os.environ.get("GPU_MEMORY_LIMIT", 2048))



mcts = MCTS(1)
def train_mcts():
    NN = NeuralNetwork()

    NN.load_checkpoint(folder='assets/data/daily_donkey', filename='save48.h5')

    game = Connect4() 
    for k in range(100):
        boardToPredict = mcts.selection(game)
        if boardToPredict != None:
            pi, v = NN.predictBatch({0 : boardToPredict}) 
            mcts.backpropagation(pi[0],v[0])         
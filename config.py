import os
from app.modules.puissance4.Games import Connect4
from app.modules.puissance4.MCTS import MCTS

from app.modules.puissance4.Models import NeuralXGBoost

SPECIFICATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
GPU_MEMORY_LIMIT = int(os.environ.get("GPU_MEMORY_LIMIT", 2048))

NN = NeuralXGBoost(use_v=False)

NN.load(xgb_folder='./assets/data/xgb', nn_folder='./assets/data', nn_filename='daily_donkey/save48.h5')

cpuct = 1
mcts = MCTS(cpuct)
game = Connect4() 

def train_mcts():
    global mcts
    for k in range(100):
        boardToPredict = mcts.selection(game)
        if boardToPredict != None:
            pi, v = NN.predictBatch({0 : boardToPredict}) 
            mcts.backpropagation(pi[0],v[0])         

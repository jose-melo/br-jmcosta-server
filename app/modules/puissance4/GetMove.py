from app.modules.puissance4.MCTS import MCTS
from .Games import Connect4
from config import NN, mcts

import numpy as np

def coupajouer(plateau):
    cpuct = 1
    mcts = MCTS(cpuct)
    for k in range(100):
        boardToPredict = mcts.selection(plateau)
        if boardToPredict != None:
            pi, v = NN.predictBatch({0 : boardToPredict}) 
            mcts.backpropagation(pi[0],v[0])         

    pi = mcts.getActionProb(plateau,temp=0)
    d = {i:e for i,e in enumerate(list(pi.keys()))}
    move = np.random.choice(list(d.keys()), p=list(pi.values()))
    move = d[move]
    return move

def get_move(actions=[0]):
    """
    Starts main function
    """
    try:
        game = Connect4() 
        player = 0
        for action in actions:
            game.push(action, player)
            player = (player + 1) %2

        coup = coupajouer(game)
    except Exception as e:
        
        del game

        raise Exception('Error')

    return coup

from .Games import Connect4
from .Models import NeuralXGBoost
from .MCTS import MCTS

import numpy as np

def politique(plateau):
    return NN.predictBatch({0:plateau})[0][0]

def coupajouer(plateau,joueur):
    global mcts, numMCTSSims
    if joueur == 1:
        plateau=plateau.mirror()


    for k in range(numMCTSSims):
        boardToPredict = mcts[joueur].selection(plateau)
        if boardToPredict != None:
            pi, v = NN.predictBatch({0 : boardToPredict}) 
            mcts[joueur].backpropagation(pi[0],v[0])         

    pi = mcts[joueur].getActionProb(plateau,temp=0)
    d = {i:e for i,e in enumerate(list(pi.keys()))}
    move = np.random.choice(list(d.keys()), p=list(pi.values()))
    move = d[move]
    return move

#Le joueur rouge commence toujours

def reverse_actions(input_str):
    
    actions = []
    last_action = int(input_str[-1])
    input_str = input_str[:-1]
    total_moves = len(input_str) - 84 + 9
    
    if total_moves < 10:
        m = -1
        for i in input_str:
            if int(i) > m:
                m = int(i)
        total_moves = m

    if total_moves % 2:
        player = 0
        actions.append((last_action, player))
    else:
        player = 1
        actions.append((last_action, 1))

    input_str = input_str.replace(str(total_moves), 'x')
    total_moves -= 1
    while total_moves > 0:
        idx = input_str.find(str(total_moves))
        input_str = input_str.replace(str(total_moves), 'x')
        player = (player + 1) % 2
        actions.insert(0, (idx // 7, player))
        total_moves -= 1

    return actions
    

def get_move(actions=[0],fen="0"*7*6*2+"0", iterations=100):
    """
    Starts main function
    """

    global numMCTSSims, NN, joueurs, mcts
    
    ### Paramètres ###

    numMCTSSims = iterations
    cpuct = 1
    mcts = [MCTS(cpuct),MCTS(cpuct)]
    try:
        NN = NeuralXGBoost(use_v=False)

        NN.load(xgb_folder='assets/data/xgb', nn_folder='assets/data', nn_filename='daily_donkey/save48.h5')

        game = Connect4() 
        player = 0
        for action in actions:
            game.push(action, player)
            player = (player + 1) %2

        game.show()
        try:
            coup = coupajouer(game,1)
        except Exception as e:
            print('[ERROR]: ', e)
            coup = 0
    except Exception as e:
        
        del NN
        del game
        del mcts

        raise Exception('Error')
    del NN
    del game
    del mcts

    return coup

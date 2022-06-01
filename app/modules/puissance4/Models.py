import os
import numpy as np
import abc
import joblib

#import tensorflow as tf
#from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Reshape, Flatten, Add, Dropout
#from tensorflow.keras.models import Model
#from tensorflow.keras import activations
#from tensorflow.keras.optimizers import Adam
import tflite_runtime.interpreter as tflite

#import xgboost as xgb

class NeuralNetwork():

    def __init__(self):
        self.interpreter = None

    def predictBatch(self, boards):
        """
        Estimation des politques et valeurs à partir d'un dictionnaire de plateaux
        """
        numpyBoards = np.asarray([boards[b].representation for b in boards])
        pi, v = self.prediction(numpyBoards)
        finalpi = dict()
        finalv = dict()
        for e,board in enumerate(boards):
            movespi=dict()
            for m in boards[board].get_legal_moves():
                if type(m)==int:
                    movespi[m] = pi[e,m]
                else:
                    movespi[m] = pi[e,m[0],m[1]]
            total = sum(movespi.values())
            if total<=1E-5:
                print("Shit")
                for move in movespi:
                    movespi[move]=1
            finalpi[board]=movespi
            finalv[board]=v[e][0]
        return finalpi, finalv

    def prediction(self, numpyBoards):
        numpyBoards = np.float32(numpyBoards)
        input_details_pi = self.interpreter_pi.get_input_details()
        output_details_pi = self.interpreter_pi.get_output_details()

        #input_shape = input_details[0]['shape']
        self.interpreter_pi.set_tensor(input_details_pi[0]['index'], numpyBoards)

        self.interpreter_pi.invoke()

        pi = self.interpreter_pi.get_tensor(output_details_pi[0]['index'])


        input_details_v = self.interpreter_v.get_input_details()
        output_details_v = self.interpreter_v.get_output_details()

        #input_shape = input_details[0]['shape']
        self.interpreter_v.set_tensor(input_details_v[0]['index'], numpyBoards)

        self.interpreter_v.invoke()

        v = self.interpreter_v.get_tensor(output_details_v[0]['index'])

        return pi,v 

    def load_checkpoint(self, folder, filename_pi, filename_v):
        filepath = os.path.join(folder, filename_pi)
        self.interpreter_pi = tflite.Interpreter(model_path=filepath)
        self.interpreter_pi.allocate_tensors()

        filepath = os.path.join(folder, filename_v)
        self.interpreter_v = tflite.Interpreter(model_path=filepath)
        self.interpreter_v.allocate_tensors()



class NeuralXGBoost():

    """
    NeuralXGBoost utilise les sorties intermédiaires du réseau de neurones comme entrées pour les XGBoost
    L'utilisation des sortie alternatives obtenues avec XGBoost est optionnelle avec les paramètres use_pi et use_v.
    """

    def __init__(self, use_pi=True, use_v=True):
        self.use_pi = use_pi
        self.use_v = use_v
        self.NN = NeuralNetwork()

        self.NN.load_checkpoint(folder='assets/data', filename='daily_donkey/save48.h5')
        if self.use_pi:
            self.xgb_pi = xgb.XGBClassifier(use_label_encoder=False, colsample_bytree = 0.5, n_estimators = 100)
        if self.use_v:
            self.xgb_v = xgb.XGBRegressor(colsample_bytree = 0.5, n_estimators = 100)
    
    def train(self, examples):
        input_boards, target_pis, target_vs = self.getInputTargets(examples)
        policyHeadLastLayer, valueHeadLastLayer = [],[]
        for i in range(1+len(input_boards)//100000):
            print(f'Step #{i}')
            data = input_boards[i*100000:min((i+1)*100000,len(input_boards)+1)]
            p, v = self.NN.model.predict(data)[2:]
            policyHeadLastLayer+=list(p)
            valueHeadLastLayer+=list(v)
        policyHeadLastLayer = np.array(policyHeadLastLayer)
        valueHeadLastLayer = np.array(valueHeadLastLayer)

        target_pi_class = []
        for i in range(len(target_pis)):
            target_pi_class.append(np.argmax(target_pis[i]))

        if self.use_pi:
            self.xgb_pi.fit(policyHeadLastLayer, target_pi_class, eval_metric='logloss')
            joblib.dump(self.xgb_pi, 'data/xgb_pi')

        if self.use_v:
            self.xgb_v.fit(valueHeadLastLayer, target_vs)
            joblib.dump(self.xgb_v, 'data/xgb_v')
        
    def prediction(self, numpyBoards):
        pi, v, policyHeadLastLayer, valueHeadLastLayer = self.NN.model.predict(numpyBoards)
        if self.use_pi:
            pi = self.xgb_pi.predict_proba(policyHeadLastLayer)
        if self.use_v:
            v = self.xgb_v.predict(valueHeadLastLayer)
            v = [[e] for e in v]
        return pi,v

    def load(self, xgb_folder, nn_folder, nn_filename):
        self.NN.load_checkpoint(folder=nn_folder, filename=nn_filename)
        self.xgb_pi = joblib.load(xgb_folder+'/xgb_pi')
        self.xgb_v = joblib.load(xgb_folder+'/xgb_v')
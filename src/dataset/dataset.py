import numpy as np
import pandas as pd
import pickle 

def dataset():
    data = pd.DataFrame(data={"X1":[0,0,1,1], "X2":[0,1,0,1], "Y":[0,1,1,0]})
    data.to_csv("/home/abhi_aiml/Keras_NN/XOR_deploy_keras/src/dataset/data.csv",index=False)
    
dataset()
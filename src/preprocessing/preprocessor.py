import numpy as np 
import pandas as pd
from src.config import config

def training_data_generator(training_data,X_train,Y_train):
    
    for e in range(config.epochs):
        for i in range(training_data.shape[0]//config.mb_size):
            
            X_train_mb = X_train[i*config.mb_size:(i+1)*config.mb_size,:]
            Y_train_mb = Y_train[i*config.mb_size:(i+1)*config.mb_size]
            
            yield X_train_mb,Y_train_mb
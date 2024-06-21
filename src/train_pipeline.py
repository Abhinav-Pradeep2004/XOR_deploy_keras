import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.models import Model

import src.preprocessing.preprocessor as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

from src.config import config

def binary_cross_entropy_loss(Y_hat,Y_true):
    
    return tf.keras.losses.binary_crossentropy(y_true=Y_true,y_pred=Y_hat)


def training_data():
    training_data = load_dataset("data.csv")
    X_train = training_data.iloc[:,0:2]
    Y_train = training_data.iloc[:,2]
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0],1)    
    
    datagen =  pp.training_data_generator(training_data,X_train,Y_train)
    
    return datagen
    
def training_loop():
    for e in range(config.epochs):
        for X_train_mb, Y_train_mb in training_data():

            with tf.GradientTape() as tape:

                Y_pred = functional_nn(X_train_mb, training=True)
                loss_func = binary_cross_entropy_loss(Y_pred,Y_train_mb)

            gradients = tape.gradient(loss_func,functional_nn.trainable_weights)
            optimizer.apply_gradients(zip(gradients,functional_nn.trainable_weights))

        print("Epoch # {}, Loss Function Value = {}".format(e+1,loss_func))
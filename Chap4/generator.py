import sys
import numpy as np

from keras.layers import  Dense, Reshape
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model

class Generator(object) :
    def __init__(self, width=28, height=28, channel=1, latent_size=100, model_type='simple'):

        if model_type =='simple':
            print("Define simple Gan")
        elif model_type == 'DCGAN' :
            print("Define DCGAN")


    def dc_model(selfs):
        return model

    def model(self, block_starting_size=128, num_blacks=4):
        return model

    def summary(self):
        print(model.summary())

    def save_model(self):


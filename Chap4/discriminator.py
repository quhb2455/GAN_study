import sys
import numpy as np

from keras.layers import  Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model

class Discriminator(object) :
    def __init__(self, width=28, height=28, channel=1, latent_szie=100, model_type='simple'):

        if model_type == 'simple' :
        elif model_type == 'DCGAN' :

    def dc_model(self):
        return model

    def model(self, block_starting_size=128, num_blocks=4):
        return model

    def summary(self):

    def save_model(self):
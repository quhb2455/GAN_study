from Chap3.gan import GAN
from Chap3.generator import Generator
from Chap3.discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, width=28, height=28, channels=1,
                 latent_size=100, epochs=50000, batch=32,
                 checkpoint=50, model_type=-1):

        self.W = width
        self.H = height
        self.C = channels

        self.epochs = epochs
        self.batch_size = batch
        self.chekpoint = checkpoint

        self.moel_type = model_type
        self.latent_space_size = latent_size

from .gan import GAN
from .generator import Generator
from .discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Trainer:
    def __init__(self, width=28, height=28, channel=1,
                 latent_size=100, epochs=50000, batch=32,
                 chekpoint=50, model_type='DCGAN', data_path=''):

        self.W = width
        self.H = height
        self.C = channel
        self.latent_size=latent_size
        self.epochs=epochs
        self.batch_size=batch
        self.checkpoint=chekpoint
        self.model_type=model_type


        self.generator = Generator(width=self.W, height=self.H, channel=self.C,
                                   latent_size=self.latent_size,
                                   model_type=self.model_type)
        self.discriminator = Discriminator(width=self.W, height=self.H, channel=self.C, model_type=self.model_type)

        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)
        self.load_npy(data_path)

    def load_npy(self, data_path, amount_of_data=0.25):
        self.X_trian = np.load(data_path)
        self.X_trian = self.X_trian[:int(amount_of_data * float(len(self.X_trian)))]
        self.X_trian = (self.X_trian.astype(np.float32) - 127.5) / 127.5
        self.X_trian = np.expand_dims(self.X_trian, axis=3)
        return

    def train(self):
        for e in range(self.epochs) :
            b = 0
            X_train_temp = deepcopy(self.X_trian)
            while len(X_train_temp) > self.batch_size:
                b = b + 1

                if self.flipCoin():
                    count_real_images=int(self.batch_size)
                    starting_index=randint(0, (len(X_train_temp) - count_real_images))
                    real_images_raw = X_train_temp[starting_index : (starting_index + count_real_images)]

                    X_train_temp = np.delete(X_train_temp, range(starting_index, starting_index + count_real_images), 0)
                    x_batch = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
                    y_batch = np.ones([count_real_images, 1])
                else :
                    latent_space_samples = self.sample_latent_space(self.batch_size)
                    x_batch = self.generator.Generator.predict(latent_space_samples)
                    y_batch = np.zeros([self.batch_size, 1])

                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

                if self.flipCoin(chance=0.9) :
                    y_generated_labels = np.ones([self.batch_size, 1])
                else :
                    y_generated_labels = np.zeros([self.batch_size, 1])

                x_latent_space_samples = self.latent_size
                generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

                print('Batch : {}, [Discriminator :: Loss : {}], [Generator :: Loss : {}]'.format(b, discriminator_loss, generator_loss))

                if b % self.checkpoint == 0 :
                    label = str(e) + '_' + str(b)
                    self.plot_checkpoint(label)

            print('Batch : {}, [Discriminator :: Loss : {}], [Generator :: Loss : {}]'.format(e, discriminator_loss, generator_loss))
            if e % self.checkpoint == 0:
                self.plot_checkpoint(e)


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

        self.model_type = model_type
        self.latent_space_size = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C,
                                   latent_size=self.latent_space_size)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator, discriminator=self.discriminator)

        self.load_MINIST()


    def load_MINIST(self,model_type=3):

        allowed_types=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if self.model_type not in allowed_types :
            print("ERROR : Only Integer Values from -1 to 9 are allowed")

        (self.X_trin, self.Y_train),(_, _) = mnist.load_data()
        if self.moel_type != -1:
            self.X_trin = self.X_trin[np.where(self.Y_train == int(self.model_type))[0]]

        self.X_trin = (np.float(self.X_trin) - 127.5) / 127.5
        self.X_trin = np.expand_dims(self.X_trin, axis=3)

        return


    def train(self):
        for e in range(self.epochs) :
            # batch
            count_real_images = int(self.batch_size/2)
            starting_index = randint(0, (len(self.X_trin) - count_real_images))
            real_images_raw = self.X_trin[starting_index : (starting_index + count_real_images)]
            x_real_images = real_images_raw.reshape( count_real_images, self.W, self.H, self.C)
            y_real_labels = np.ones([count_real_images, 1])

            # 나머지 batch
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.Generator.predict(latent_space_samples)
            y_generated_labels = np.zeros([self.batch_size - count_real_images, 1])

            # discriminator를 위해 data concat
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # train the discriminator using batch
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

            # train the generator
            x_latent_space_samples = self.sample_latent_space(self.batch_size)
            y_generated_labels = np.ones([self.batch_size, 1])
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            print('Epoch : ' + str(int(e)) + ', [Discriminator :: Loss : '
                  + str(discriminator_loss) + '], [Generator :: Loss : ' + str(generator_loss) + ']')

            if e % self.chekpoint == 0 :
                self.plot_checkpoint(e)

            return

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.latent_space_size))

    def plot_checkpoint(self, e):
        filename = "./model/sample_" + str(e) + ".png"

        # generator가 그린 그림을 보여줌
        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]) :
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close('all')
            return
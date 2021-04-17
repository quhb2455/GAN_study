from .gan import GAN
from .generator import Generator
from .discriminator import Discriminator
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

        self.EPOCHS = epochs
        self.batch_size = batch
        self.chekpoint = checkpoint

        self.model_type = model_type
        self.latent_space_size = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C,
                                   latent_size=self.latent_space_size)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(discriminator=self.discriminator.Discriminator, generator=self.generator.Generator)

        self.load_MINIST()


    def load_MINIST(self,model_type=3):

        allowed_types=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if self.model_type not in allowed_types :
            print("ERROR : Only Integer Values from -1 to 9 are allowed")

        # X_train : image, Y_train : label
        (self.X_train, self.Y_train),(_, _) = mnist.load_data()
        if self.model_type != -1:
            # 원하는 label 그러니까 여기선 1 ~ 9 사이의 숫자 중 하나만 선정하여 학습하기 위함.
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)

        return


    def train(self):
        for e in range(self.EPOCHS) :
            # 총 32 batch size에서 16만 사용한 batch
            count_real_images = int(self.batch_size/2)
            # 0 ~ (len(self.X_train) - count_real_images) 사이의 랜덤 숫자 1개 생성
            starting_index = randint(0, (len(self.X_train) - count_real_images))
            # 생성된 랜덤 숫자부터 (생성된 랜덤 숫자+batch)사이에 있는 value들를 real_image_raw에 넣음
            real_images_raw = self.X_train[starting_index : (starting_index + count_real_images)]
            # real_images_raw의 모양을 (batch, w, h, c)로 바꿈
            x_real_images = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
            # (batch, 1) 크기의 1로만 채워진 list 생성, real_image는 진짜기 때문에 label이 1
            y_real_labels = np.ones([count_real_images, 1])

            # 나머지 16개 batch로 latent space size를 정의해줌.
            latent_space_samples = self.sample_latent_space(count_real_images)
            # (16,100)크기의 정규 분포를 가진 latent space를 generator에 넣어서 (28, 28, 1)크기의 image 생성
            x_generated_images = self.generator.Generator.predict(latent_space_samples)
            # (16, 1)크기의 0으로만 가득찬 list를 만들어서 label에 넣는다. generated image는 가짜기 때문에 0
            y_generated_labels = np.zeros([self.batch_size - count_real_images, 1])

            # discriminator를 위해 data concat
            # x_real_images(16, 28, 28, 1)과 x_generated_images(16, 28, 28, 1)??을 합침. y도 마찬가지
            # 합치면 32, 28 , 28, 1 이렇게 됨
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # train the discriminator using batch
            # train_on_batch는 batch 단위로 학습 할 거라는 뜻. 참고 : https://www.youtube.com/watch?v=8W977CTNaEo
            # 여기선 batch가 32이 이기 때문에 for을 통해서 한번에 model 16개의 data를 넣겠다~ 이런 의미.
            # GAN에서 많이 쓰는 학습이래
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

            # train the generator
            # (32, 100)의 latent space를 만듬
            x_latent_space_samples = self.sample_latent_space(self.batch_size)
            # generated label (32, 1)을 1로 채움
            y_generated_labels = np.ones([self.batch_size, 1])
            # gan model = generator + discriminator 임
            # input은 generator에 맞게 넣고 output은 discriminator에 맞게 나옴.
            # 나오는 loss은 generator가 얼마나 학습 했냐에 대한 loss 임.
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            print('Epoch : ' + str(int(e)) + ', [Discriminator :: Loss : '
                  + str(discriminator_loss) + '], [Generator :: Loss : ' + str(generator_loss) + ']')

            # 50 epoch 마다 학습 중인 generator를 이용해서 img를 생성하여 저장.
            if e % self.chekpoint == 0 :
                self.plot_checkpoint(e)

        return

    def sample_latent_space(self, instances):
        # 정규분포를 random하게 채움. size = (16, latent_space_size=100)
        # 1개의 batch에 100 크기의 latent space 할당
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

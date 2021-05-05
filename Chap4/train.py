from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import cv2

class Trainer:
    def __init__(self, width=28, height=28, channel=1,
                 latent_size=100, epochs=50000, batch=32,
                 checkpoint=50, model_type='DCGAN', data_path=''):

        self.W = width
        self.H = height
        self.C = channel
        self.latent_size=latent_size
        self.epochs=epochs
        self.batch_size=batch
        self.checkpoint=checkpoint
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
        print(self.X_trian.shape)
        # self.X_trian = np.expand_dims(self.X_trian, axis=3)
        # return

    def flipCoin(self, chance=0.5):
        # 0, 1을 출력하는데 1이 나올 확률을 chance 만큼 주겠다는 의미
        return np.random.binomial(1, chance)

    def sample_latent_space(self, instance):
        # 평균이 0 이고 표준편차가 1인 (instance , latent_size)의 크기를 가진 정규분포 생성
        return np.random.normal(0, 1, (instance, self.latent_size))

    def save_result(self, name):
        latent_space_samples = self.sample_latent_space(self.batch_size)
        img = self.generator.Generator.predict(latent_space_samples)
        ran = randint(0, 128)
        cv2.imwrite('./results/'+name+'.jpg',img[ran])

    def train(self):
        for e in range(self.epochs) :

            # b = batch
            b = 0
            e_start_t = time.time()

            # training data 복사본 생성
            X_train_temp = deepcopy(self.X_trian)

            while len(X_train_temp) > self.batch_size:

                # batch를 증가 시킴
                b = b + 1
                b_start_t = time.time()

                if self.flipCoin():
                    count_real_images=int(self.batch_size)

                    # 0 ~ len(X_train_temp)-count_real_images 사이에서 1개의 랜덤 숫자를 starting_index에 넣음
                    starting_index=randint(0, (len(X_train_temp) - count_real_images))

                    # random숫자부터 ~ batch_Size만큼을 real_images_raw에 저장
                    real_images_raw = X_train_temp[starting_index : (starting_index + count_real_images)]

                    # X_train_temp에서 range에 속한 범위의 숫자를 지움. 0은 차원을 의미
                    X_train_temp = np.delete(X_train_temp, range(starting_index, starting_index + count_real_images), 0)

                    # 위에서 X_train_temp의 일정 부분을 가져온 real_images_raw의 shape을 (batch, w, h, c)모양으로 바꿈
                    x_batch = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)

                    # [count_real_images, 1]크기의 1로 가득찬 np array생성
                    y_batch = np.ones([count_real_images, 1])


                else :

                    # (batch_size, latent_size)크기의 평균이 0 표준편차가 1인 정규분포를 따르는 latent space 생성
                    latent_space_samples = self.sample_latent_space(self.batch_size)

                    # 정규분포 latent space를 넣어서 fake image 생성
                    x_batch = self.generator.Generator.predict(latent_space_samples)

                    # [batch_size, 1] 크기의 0으로 가득찬 np array 생성
                    y_batch = np.zeros([self.batch_size, 1])

                # fake image와 0(가짜)로 가득찬 label을 넣어서 discriminator 학습
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

                # 10퍼센트의 확률로 틀린 label을 생성해서 학습에 사용하는데 이러면 수렴 속도에 차이가 있다고 한다.
                if self.flipCoin(chance=0.9) :
                    y_generated_labels = np.ones([self.batch_size, 1])
                else :
                    y_generated_labels = np.zeros([self.batch_size, 1])

                x_latent_space_samples = self.sample_latent_space(self.batch_size)
                generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

                # batch time check
                b_end_t = time.time() - b_start_t
                print('Batch : {}, [Discriminator :: Loss : {}], [Generator :: Loss : {}], [time :: {}]'.format(b, discriminator_loss, generator_loss, b_end_t))

                if b % self.checkpoint == 0 :
                    label = str(e) + '_' + str(b)
                    self.save_result(label)
            e_end_t = time.time() - e_start_t
            print()
            print('Epoch : {}, [Discriminator :: Loss : {}], [Generator :: Loss : {}], [time :: {}]'.format(e, discriminator_loss, generator_loss, e_end_t))
            print()
            # if e % self.checkpoint == 0:
            #     self.plot_checkpoint(e)


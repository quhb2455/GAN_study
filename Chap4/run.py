from train import Trainer

if __name__ == "__main__":

    # predefine image64 size
    HEIGHT = 64
    WIDTH = 64
    CHANNEL = 3

    # latent space
    LATENT_SPACE_SIZE = 100

    # train
    EPOCHS = 100
    BATCH = 128
    CHECKPOINT = 50
    PATH = "./church_outdoor_train_lmdb_color.npy"
    # MODEL_TYPE = -1

    trainer = Trainer(height=HEIGHT,
                      width=WIDTH,
                      channel=CHANNEL,
                      latent_size=LATENT_SPACE_SIZE,
                      epochs=EPOCHS,
                      batch=BATCH,
                      checkpoint=CHECKPOINT,
                      model_type='DCGAN',
                      data_path=PATH)

    trainer.train()
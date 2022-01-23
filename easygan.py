import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


class GAN():
    def __init__(self, batch_size, image_width, image_height, image_channels):
        self.batch_size = batch_size
        self.width = image_width
        self.height = image_height
        self.channels = image_channels

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
        self.discriminator.trainable = False
        combine_in = keras.layers.Input(shape=(20))
        generated = self.generator(combine_in)
        combine_out = self.discriminator(generated)
        self.combined = keras.Model(combine_in, combine_out)
        self.combined.compile(optimizer="adam", loss="binary_crossentropy")

    def build_generator(self):
        generator = keras.models.Sequential([
            keras.layers.Dense(256, input_dim=20),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(1024),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(np.prod((self.width, self.height, self.channels)), activation="sigmoid"),
            keras.layers.Reshape((self.width, self.height, self.channels))
        ])

        return generator

    def build_discriminator(self):
        discriminator = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(self.width, self.height, self.channels)),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        return discriminator

    def train(self, epochs, data, rows, columns):
        for epoch in range(epochs):
            fig = plt.figure()
            epoches = 0
            for images in data:
                seed = np.random.normal(0, 1, (self.batch_size, 20))
                images = images.reshape(self.batch_size, self.width, self.height, self.channels)

                fake_images = self.generator.predict(seed)

                fake_images = fake_images.reshape(self.batch_size, self.width, self.height, self.channels)

                self.discriminator.train_on_batch(fake_images, np.zeros((self.batch_size, 1)))
                self.discriminator.train_on_batch(images, np.ones((self.batch_size, 1)))
                self.combined.train_on_batch(seed, np.ones((self.batch_size, 1)))
                epoches += 1
                print(epoches)
                if epoches % 100 == 0:
                    for n in range(self.batch_size):
                        img = fake_images[n]
                        if self.channels == 3:
                            img = img.reshape(self.width, self.height, self.channels)
                        else:
                            img = img.reshape(self.width, self.height)
                        plt.subplot(rows, columns, n + 1)
                        plt.imshow(img)

                    plt.draw()
                    plt.pause(0.1)
                    fig.clear()

                    
                    
class rand_layer(keras.layers.Layer):
    def call(self, inputs):
        return inputs*tf.random.uniform((1,),0.9,1)

class TipGAN():
    def __init__(self) -> None:
        try:
            shutil.rmtree("downloads")
        except Exception:
            pass
        self.model = self.build_model()
        self.model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    def build_model(self):
        model = keras.models.Sequential([
            keras.layers.Input(shape=(100)),
            keras.layers.Dense(400,activation="relu"),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(2500,activation="sigmoid"),
            rand_layer(),
            keras.layers.Reshape((50,50)),            
        ])

        return model

    def load_images_from_folder(self,folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(50,50))
            if img is not None:
                images.append(img)
        return np.array(images).astype(np.float32)/255

    def draw(self,text,epochs=10,interval= 10,save=False):
        try:
            shutil.rmtree(".//out")
            os.mkdir(".//out")
        except Exception:
            pass
        response = google_images_download.googleimagesdownload()
        arguments = {"keywords":text,"limit":5,"print_urls":True}
        paths = response.download(arguments)
        data = self.load_images_from_folder(".//downloads//"+text)
        for i in range(epochs):
            print(i)
            seed = np.random.normal(0.9,1,(5,100))
            self.model.train_on_batch(seed,data)
            if i % interval == 0:
                seed2 = np.random.normal(0.9,1,(1,100))
                img = self.model.predict(seed2).reshape(50,50)
                plt.imshow(img,cmap="binary")
                plt.imsave(".//out//"+str(i//interval)+".png",img)
                plt.pause(0.1)                   

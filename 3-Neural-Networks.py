import os
import tensorflow as tf
import keras
from keras import losses
from keras import optimizers
from keras import layers
from keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Custom_Network:
    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=(28*28)),
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(10),
            ]
        )
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
        )

    def getModel(self):
        return self.model


def load_Data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_Data()
    networkk = Custom_Network()
    model = networkk.getModel()
    print(model.summary())
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1)
    model.save("model.h5")
    model.evaluate(x_test, y_test, batch_size=64, verbose=1)


if __name__ == '__main__':
    main()

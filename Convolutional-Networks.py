import os
import tensorflow as tf
import keras
from keras import layers
from keras import activations
from keras.datasets import cifar10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class ConvNet:
    def __init__(self):
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3)(inputs)
        x = layers.BatchNormalization()(x)
        x = activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3)(x)
        x = layers.BatchNormalization()(x)
        x = activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3)(x)
        x = layers.BatchNormalization()(x)
        x = activations.relu(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(10)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            metrics=['accuracy'],
        )

    def getModel(self):
        return self.model


def load_Data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_Data()
    Net = ConvNet()
    model = Net.getModel()
    print(model.summary())
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1)
    model.save("ConvModel.h5")
    model.evaluate(x_test, y_test, batch_size=128, verbose=1)


if __name__ == '__main__':
    main()
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Global variables
IMG_SHAPE = (32, 32, 3)

def build_model_cnn3():
    """CNN3

    Notes
    -----
    [description]
    """
    # 读取数据集
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 建立模型, Block 1
    model = models.Sequential()
    model.add(layers.Conv2D(96, (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model.add(layers.Conv2D(96, (3, 3), activation='relu'))
    model.add(layers.Conv2D(96, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(192, (3, 3), activation='relu'))
    model.add(layers.Conv2D(192, (3, 3), activation='relu'))
    model.add(layers.Conv2D(192, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(10, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())

    # Output
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(
        optimizer="adam",\
        #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
        loss = 'categorical_crossentropy',\
        metrics=["accuracy"]
    )
    # 将标签转化为one-hot
    num_classes = len(np.unique(train_labels))
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes)
    y_test = tf.keras.utils.to_categorical(test_labels, num_classes)
    model.fit(train_images, y_train, epochs=50, batch_size=128, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, y_test, verbose=2)
    print(test_acc)

if __name__ == '__main__':
    build_model_cnn3()
    build_model_cnn3().save('E:/Models/CNN3/model3.h5')
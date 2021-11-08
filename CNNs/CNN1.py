import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Global variables
IMG_SHAPE = (32, 32, 3)
# IMG_HEIGHT =32
# IMG_WIDTH = 32
# batch_size = 256
# epochs = 20
#
# def model1():
#     input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
#
#     # block1
#     conv1 = Conv2D(64, 3, padding='same', activation='relu')(input)
#     conv2 = Conv2D(64, 3, padding='same', activation='relu')(conv1)
#     pool1 = MaxPooling2D()(conv2)
#
#     # block2
#     conv3 = Conv2D(128, 3, padding='same', activation='relu')(pool1)
#     conv4 = Conv2D(128, 3, padding='same', activation='relu')(conv3)
#     pool2 = MaxPooling2D()(conv4)
#
#     flatten = Flatten()(pool2)
#     dense1 = Dense(256)(flatten)
#     dense2 = Dense(256)(dense1)
#     predict = Dense(10, activation = 'softmax')(dense2)
#
#     model=Model(inputs = input, outputs=predict)
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#
#     return model
# model = model1()
#
# def load_pickle(f):
#     version = platform.python_version_tuple() # 取python版本号
#     if version[0] == '2':
#         return  pickle.load(f) # pickle.load, 反序列化为python的数据类型
#     elif version[0] == '3':
#         return  pickle.load(f, encoding='latin1')
#     raise ValueError("invalid python version: {}".format(version))
#
# def load_CIFAR_batch(filename):
#     """ load single batch of cifar """
#     with open(filename, 'rb') as f:
#         datadict = load_pickle(f)  # dict类型
#         X = datadict['data']  # X, ndarray, 像素值
#         Y = datadict['labels']  # Y, list, 标签, 分类
#
#         # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
#         # transpose，转置
#         # astype，复制，同时指定类型
#         X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
#         Y = np.array(Y)
#         return X, Y
#
# def load_CIFAR10(ROOT):
#     """ load all of cifar """
#     xs = []  # list
#     ys = []
#
#     # 训练集batch 1～5
#     for b in range(1, 6):
#         f = os.path.join(ROOT, 'data_batch_%d' % (b,))
#         X, Y = load_CIFAR_batch(f)
#         xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
#         ys.append(Y)
#     Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
#     Ytr = np.concatenate(ys)
#     del X, Y
#
#     # 测试集
#     Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#     return Xtr, Ytr, Xte, Yte
#
#
# #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# #x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train, y_train, x_test, y_test = load_CIFAR10('D:/PyCharm/pythonProject/Repair/data/cifar-10-batches-py')
#
#
# # 将标签转化为one-hot
# num_classes = len(np.unique(y_train))
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)
#
# #print(x_train)
# model.fit(x_train, y_train,epochs = 20,batch_size =256, validation_data = (x_test, y_test))
# test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
# print(test_acc)
#
# # checkpointer = ModelCheckpoint(filepath='E:/Models/CNN1.hdf5', verbose=1,
# #                                save_best_only=True)
#
# model.save('E:/Models/CNN1/model1.h5')
from tensorflow.python.tpu import datasets


def build_model_cnn1():
    """CNN1

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
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3: empty

    # Output
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Dense(256))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(
        optimizer="adam",\
        loss='categorical_crossentropy', \
        metrics=["accuracy"]
    )
    # 将标签转化为one-hot
    num_classes = len(np.unique(train_labels))
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes)
    y_test = tf.keras.utils.to_categorical(test_labels, num_classes)
    model.fit(train_images, y_train, epochs=125, batch_size=64, validation_data=(test_images, y_test))
    test_loss, test_acc = model.evaluate(test_images, y_test, verbose=2)
    print(test_acc)
    build_model_cnn1().save('E:/Models/CNN1/model1.h5')

if __name__ == '__main__':
    build_model_cnn1()


'''
Author: Radon
Date: 2021-10-28 12:08:04
LastEditors: Radon
LastEditTime: 2021-11-08 18:08:50
Description: Hi, say something
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlwt
import traceback
import os
import shutil
import cv2

from tensorflow.keras import datasets, layers, models, losses


def train():
    """根据Google的官方教程搭建一个简单的CNN模型

    Notes
    -----
    [description]
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Build CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))
    model.summary()

    model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("test_loss: %f, test_acc: %f" % (test_loss, test_acc))
    model.save("CNNExample.h5")


def get_my_style() -> xlwt.XFStyle:
    """设置向excel文件写数据时的格式

    Returns
    -------
    xlwt.XFStyle
        style

    Notes
    -----
    [description]
    """
    style = xlwt.XFStyle()

    font = xlwt.Font()
    font.name = "Times New Roman"

    style.font = font

    return style


def get_diff():
    """加载模型，输出预测结果与实际结果之间差异的详细内容，并输出图片

    Notes
    -----
    [description]
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    images, labels = train_images, train_labels  # 用训练集的图片和标签进行验证

    # 使用模型对测试集进行预测，并统计数据
    model = models.load_model("CNNExample.h5")  # TODO: 加载模型, 可以在这里更换要加载的模型
    # model.evaluate(images, labels, verbose=2)
    predictions = model.predict(images, batch_size=32, verbose=1)  # batch_size是否必要?
    predict_result = np.argmax(predictions, axis=1)  # 获取测试集图像的预测结果
    result_arr = np.zeros((10, 10), dtype=int)  # 标签有10个, 因此创建一个10*10, 初值为0的数组

    if os.path.exists("faultFigs"):  # 创建存储错误分类图像的文件夹
        shutil.rmtree("faultFigs")
    while True:
        try:
            os.mkdir("faultFigs")
            break
        except PermissionError:
            print("Damn it, folder create failed because permission denied, try again")

    cnt = 0
    first_r1p0_img, first_r0p1_img = False, False
    for i in range(len(predict_result)):  # 查看预测结果与实际结果的差别
        if predict_result[i] != labels[i][0]:
            cnt += 1
            # 将预测错误的图片保存到同目录下的faultFigs文件夹下, 格式为 实际_预测_计数.jpg
            if not first_r0p1_img and labels[i][0] == 0 and predict_result[i] == 1:  # 将第一张真实为0预测为1的图片保存到文件夹下
                cv2.imwrite(os.path.join("faultFigs", "R{:d}_P{:d}_C{:d}.jpg".format(labels[i][0], predict_result[i], cnt)), images[i] * 255.0)
                idx_01 = i  # 记录第一张R0P1图片的下标
                first_r0p1_img = True
            if not first_r1p0_img and labels[i][0] == 1 and predict_result[i] == 0:  # 将第一张真实为1预测为0的图片保存到文件夹下
                cv2.imwrite(os.path.join("faultFigs", "R{:d}_P{:d}_C{:d}.jpg".format(labels[i][0], predict_result[i], cnt)), images[i] * 255.0)
                idx_10 = i  # 记录第一张R1P0图片的下标
                first_r1p0_img = True
        result_arr[labels[i][0]][predict_result[i]] += 1
    print("不准确率:", cnt / len(predict_result), ", 请确认不准确率与准确率的和为1")

    # 将两张图片的第一个像素进行交换
    pixel1 = images[idx_01][0][0]  # 获取R0P1的第一个像素
    pixel2 = images[idx_10][0][0]  # 获取R1P0的第一个像素
    images[idx_01][0][0] = pixel2  # 将R0P1的第一个像素进行替换
    images[idx_10][0][0] = pixel1  # 将R1P0的第一个像素进行替换
    cv2.imwrite(os.path.join("faultFigs", "R0P1_switch.jpg"), images[idx_01] * 255.0)
    cv2.imwrite(os.path.join("faultFigs", "R1P0_switch.jpg"), images[idx_10] * 255.0)

    # 将统计数据输出至xls文件
    try:
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Sheet")
        my_style = get_my_style()
        sheet.write(0, 0, "Real\\Pred", my_style)  # 设置表头
        for i in range(10):
            sheet.write(0, i + 1, i, my_style)  # 设置第一行，预测结果的编号
            sheet.write(i + 1, 0, i, my_style)  # 设置第一列，实际结果的编号
            for j in range(10):
                sheet.write(i + 1, j + 1, int(result_arr[i][j]), my_style)
        workbook.save("test_result.xls")
    except:
        print("\033[1;31m保存失败，可能还开着xls文件")
        traceback.print_exc()
        print("\033[0m")


if __name__ == '__main__':
    # train()
    get_diff()
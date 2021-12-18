'''
Author: Radon
Date: 2021-12-18 14:41:59
LastEditors: Radon
LastEditTime: 2021-12-18 15:39:52
Description: Hi, say something
'''
import cv2
import random
import math
import numpy as np


def eraseImg(img, type="mess", p=0.5, sl=0.02, sh=0.4, r1=0.3):
    """随机擦除图片的一部分

    Parameters
    ----------
    img : array
        通过imread读取的图片
    type : str, optional (defaults = "mess")
        擦除方式，擦除区域全黑(black)，全白(white)或随机(mess)
    p : float, optional (defaults = 0.5)
        擦除概率
    sl : float, optional (defaults = 0.02)
        min erasing area region
    sh : float, optional (defaults = 0.4)
        max erasing area region
    r1 : float, optional (defaults = 0.3)
        min aspect ratio range of earsing region

    Returns
    -------
    array
        擦除后的图片

    Notes
    -----
    [description]
    """
    if random.random() > p: # 生成0-1之间的数，若大于p则表示不擦除，即p越大擦除的概率越大
        return img

    s = (sl, sh)
    r = (r1, 1 / r1)
    while True:
        se = random.uniform(*s) * img.shape[0] * img.shape[1]
        re = random.uniform(*r)

        he = int(round(math.sqrt(se * re)))
        we = int(round(math.sqrt(se / re)))

        xe = random.randint(0, img.shape[1])
        ye = random.randint(0, img.shape[0])

        if (xe + we <= img.shape[1] and ye + he <= img.shape[0]):
            if type == "black":
                img[ye:ye + he, xe:xe + we, :] = np.random.randint(low=0, high=1, size=(he, we, img.shape[2]))
            elif type == "white":
                img[ye:ye + he, xe:xe + we, :] = np.random.randint(low=255, high=256, size=(he, we, img.shape[2]))
            else:
                img[ye:ye + he, xe:xe + we, :] = np.random.randint(low=0, high=255, size=(he, we, img.shape[2]))
            return img


def main():
    img = cv2.imread("test.png")
    for i in range(5):
        imgTemp = eraseImg(img.copy(), p=1)
        cv2.imshow("temp", imgTemp)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
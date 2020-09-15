
from PIL import Image
import numpy as np
import random
import math


def cal_dis(pA, pB):
    return math.sqrt((pA[0] - pB[0])**2 + (pA[1] - pB[1])**2)


def add_noise(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R = random.randint(1, 3)
    P_noise_x = random.randint(R, w - R)
    P_noise_y = random.randint(R, h - R)
    for i in range(h):
        for j in range(w):
            if cal_dis((i, j), (P_noise_x, P_noise_y)) < R:
                if random.random() < 0.8:
                    img[i][j] = 0
    R = random.randint(1, 6)
    P_noise_x = random.randint(R, w - R)
    P_noise_y = random.randint(R, h - R)
    for i in range(h):
        for j in range(w):
            if P_noise_x < j < P_noise_x + R and P_noise_y < i < P_noise_y + R:
                if random.random() < 0.8:
                    img[i][j] = 0
    img = Image.fromarray(img)
    return img


if __name__ == '__main__':
    img = Image.open('data/book_pages/imgs_vertical/book_page_0.jpg')
    img = add_noise(img)
    img.show()
